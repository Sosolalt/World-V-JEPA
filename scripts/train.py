"""V-JEPA training entry point."""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_vjepa.dataset import BilliardDataset
from mini_vjepa.ema import tau_schedule
from mini_vjepa.vjepa import VJEPA, count_parameters


def load_config(path: str) -> Mapping[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def split_param_groups(model: VJEPA, wd_encoder: float, wd_predictor: float) -> list[dict]:
    encoder_params = list(model.encoder.parameters())
    predictor_params = (
        list(model.predictor.parameters())
        + [model.temporal_pos, model.mask_token]
    )
    encoder_ids = {id(p) for p in encoder_params}
    predictor_ids = {id(p) for p in predictor_params}
    # Sanity: target_encoder params have requires_grad=False, exclude.
    for p in model.target_encoder.parameters():
        if p.requires_grad:
            raise RuntimeError("target_encoder params must not require grad")
    # Verify partition covers all trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    covered = encoder_ids | predictor_ids
    for p in trainable:
        if id(p) not in covered:
            raise RuntimeError("trainable parameter not assigned to a group")
    return [
        {"params": [p for p in encoder_params if p.requires_grad], "weight_decay": wd_encoder},
        {"params": [p for p in predictor_params if p.requires_grad], "weight_decay": wd_predictor},
    ]


def make_lr_lambda(warmup_steps: int, total_steps: int, lr_peak: float, lr_final: float):
    # LambdaLR multiplies the base LR (= lr_peak). We return a scale in [0, 1].
    final_scale = lr_final / lr_peak if lr_peak > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        decay_steps = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_scale + (1.0 - final_scale) * cos

    return lr_lambda


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--data", default=None, type=str, help="Override NPZ dataset path.")
    p.add_argument("--epochs", default=None, type=int, help="Override training.epochs.")
    p.add_argument("--debug", action="store_true",
                   help="Run 2 epochs on data/sample/sample.npz regardless of config.")
    p.add_argument("--baseline", default=None, choices=["pixel"],
                   help="Train a baseline instead of V-JEPA (currently: 'pixel').")
    p.add_argument("--override", action="append", default=[],
                   metavar="key.path=value",
                   help="Override a nested config value. Repeatable. Value is "
                        "coerced to int/float/bool/str. Example: "
                        "--override training.cov_loss_weight=25 "
                        "--override training.reg_target=context")
    p.add_argument("--tag", default=None, type=str,
                   help="Suffix appended to the run directory name (helps "
                        "label experiments).")
    p.add_argument("--lr-schedule-epochs", default=None, type=int,
                   help="If set, LR cosine schedule is computed over this many "
                        "epochs while the run still terminates after --epochs. "
                        "Use to preview the first N epochs of a long run "
                        "without running the full duration.")
    return p.parse_args()


def _coerce(value: str):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"--override expects key=value, got {spec!r}")
        key_path, raw_value = spec.split("=", 1)
        keys = key_path.split(".")
        d = cfg
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = _coerce(raw_value)
    return cfg


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    if args.debug:
        data_path = str(REPO_ROOT / "data" / "sample" / "sample.npz")
        epochs = 2
    else:
        data_path = args.data if args.data else cfg["data"]["output_path"]
        epochs = int(args.epochs) if args.epochs is not None else int(cfg["training"]["epochs"])

    if args.baseline == "pixel":
        from baselines.pixel_predictor import train_pixel_baseline
        return train_pixel_baseline(cfg, data_path, epochs, REPO_ROOT)

    set_seed(int(cfg["training"]["seed"]))
    device = select_device()
    print(f"[train] device={device}, data={data_path}, epochs={epochs}", flush=True)

    dataset = BilliardDataset(data_path)
    if len(dataset) == 0:
        raise RuntimeError(f"dataset is empty: {data_path}")
    batch_size = int(cfg["training"]["batch_size"])
    effective_bs = min(batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=effective_bs,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    model = VJEPA(cfg["model"]).to(device)
    counts = count_parameters(model)
    print(f"[train] params total={counts['total']:,} trainable={counts['trainable']:,}",
          flush=True)

    # Optimizer with per-group weight decay
    param_groups = split_param_groups(
        model,
        wd_encoder=float(cfg["training"]["weight_decay_encoder"]),
        wd_predictor=float(cfg["training"]["weight_decay_predictor"]),
    )
    lr_peak = float(cfg["training"]["lr_peak"])
    lr_final = float(cfg["training"]["lr_final"])
    optimizer = AdamW(param_groups, lr=lr_peak)

    # Step-wise LR schedule (warmup over warmup_epochs * steps_per_epoch, then cosine to lr_final)
    steps_per_epoch = max(1, math.ceil(len(dataset) / effective_bs))
    warmup_epochs = int(cfg["training"]["warmup_epochs"])
    warmup_steps = warmup_epochs * steps_per_epoch
    schedule_epochs = args.lr_schedule_epochs if args.lr_schedule_epochs is not None else epochs
    total_steps = schedule_epochs * steps_per_epoch
    scheduler = LambdaLR(optimizer, lr_lambda=make_lr_lambda(warmup_steps, total_steps, lr_peak, lr_final))

    sequence_length = int(cfg["data"]["sequence_length"])
    context_frames = int(cfg["model"]["context_frames"])
    delta_t_min = int(cfg["model"]["delta_t_min"])
    delta_t_max = int(cfg["model"]["delta_t_max"])
    var_weight = float(cfg["training"]["var_loss_weight"])
    cov_weight = float(cfg["training"].get("cov_loss_weight", 1.0))
    reg_target = str(cfg["training"].get("reg_target", "z_pred"))
    loss_fn_name = str(cfg["training"].get("loss_fn", "mse"))
    mask_ratio = float(cfg["training"].get("mask_ratio", 0.0))
    grad_clip = float(cfg["training"]["grad_clip"])
    grad_value_clip = float(cfg["training"].get("grad_value_clip", 0.0))
    checkpoint_every = int(cfg["training"]["checkpoint_every"])
    max_checkpoints = int(cfg["training"]["max_checkpoints"])
    ema_tau_start = float(cfg["training"]["ema_tau_start"])
    ema_tau_end = float(cfg["training"]["ema_tau_end"])
    collapse_eff_rank_threshold = float(
        cfg["training"].get("collapse_eff_rank_threshold", 3.0)
    )
    collapse_patience = int(cfg["training"].get("collapse_patience_epochs", 10))

    max_t_start = sequence_length - context_frames - delta_t_max
    if max_t_start <= 0:
        raise ValueError(
            f"sequence_length={sequence_length} too short for context_frames={context_frames} + delta_t_max={delta_t_max}"
        )

    # Run dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dirname = timestamp if not args.tag else f"{timestamp}_{args.tag}"
    run_dir = REPO_ROOT / "runs" / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    # Persist the effective config (post-overrides) so experiments are reproducible.
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(
        f"[train] reg_target={reg_target} loss_fn={loss_fn_name} "
        f"mask_ratio={mask_ratio} cov_w={cov_weight} var_w={var_weight} "
        f"ema_tau_start={ema_tau_start}",
        flush=True,
    )
    metrics_path = run_dir / "metrics.csv"
    metrics_columns = [
        "epoch", "lr", "tau", "loss", "mse", "var_reg", "cov_reg",
        "avg_std", "effective_rank", "avg_cosine_sim",
        "ctx_avg_std", "ctx_effective_rank", "ctx_avg_cosine_sim",
        "time_s",
    ]
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(metrics_columns)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Checkpoint bookkeeping
    saved_ckpts: list[Path] = []
    best_eff_rank = -float("inf")
    best_ckpt_path: Path | None = None
    metrics_history: list[dict] = []
    low_std_streak = 0
    collapse_streak = 0

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        tau = tau_schedule(epoch, schedule_epochs, start=ema_tau_start, end=ema_tau_end)

        running_loss = 0.0
        running_mse = 0.0
        running_var = 0.0
        running_cov = 0.0
        n_batches = 0

        last_z_pred_detached: torch.Tensor | None = None
        last_ctx_detached: torch.Tensor | None = None

        for batch in loader:
            frames_u8 = batch["frames"].to(device, non_blocking=False)  # (B, T, 3, H, W)
            frames = frames_u8.float() / 255.0
            b = frames.shape[0]

            t_start = random.randint(0, max_t_start)
            delta_t = random.randint(delta_t_min, delta_t_max)

            context = frames[:, t_start: t_start + context_frames].contiguous()
            target_frame = frames[:, t_start + context_frames - 1 + delta_t].contiguous()
            dt_tensor = torch.full((b,), delta_t, device=device, dtype=torch.long)

            context_tokens = model.encode_context(context)
            # Apply mask token replacement to a fraction of context positions
            # before the predictor (mask_ratio=0.0 returns context unchanged).
            predictor_input = model.apply_context_mask(context_tokens, mask_ratio)
            z_pred = model.predict(predictor_input, dt_tensor)
            z_target = model.target_encode(target_frame)

            out = model.compute_loss(
                z_pred,
                z_target,
                context_tokens=context_tokens,
                var_weight=var_weight,
                cov_weight=cov_weight,
                reg_target=reg_target,
                loss_fn=loss_fn_name,
            )
            loss = out["loss"]

            # NaN-abort. A single bad batch can produce a non-finite loss
            # and from there the loop will "train" with NaN parameters for
            # the rest of the run, wasting compute and masking the failure.
            # Cheaper to bail immediately and surface the failure to caller.
            if not torch.isfinite(loss):
                print(
                    f"[train] non-finite loss at epoch {epoch} batch {n_batches}; aborting run",
                    file=sys.stderr, flush=True,
                )
                return 2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(trainable_params, grad_clip)
            if grad_value_clip > 0:
                clip_grad_value_(trainable_params, grad_value_clip)
            optimizer.step()
            scheduler.step()
            model.ema_update(tau)

            running_loss += float(loss.detach().item())
            running_mse += float(out["mse"].item())
            running_var += float(out["var_reg"].item())
            running_cov += float(out["cov_reg"].item())
            n_batches += 1
            last_z_pred_detached = z_pred.detach()
            last_ctx_detached = context_tokens.detach()

        avg_loss = running_loss / max(1, n_batches)
        avg_mse = running_mse / max(1, n_batches)
        avg_var = running_var / max(1, n_batches)
        avg_cov = running_cov / max(1, n_batches)

        # Compute heavy monitors once per epoch on the last training batch's z_pred.
        # effective_rank may fall back to CPU on MPS (PYTORCH_ENABLE_MPS_FALLBACK=1).
        model.eval()
        with torch.no_grad():
            if last_z_pred_detached is None:
                avg_std_v = 0.0
                eff_rank_v = 0.0
                cos_sim_v = 0.0
                ctx_std_v = 0.0
                ctx_rank_v = 0.0
                ctx_cos_v = 0.0
            else:
                from mini_vjepa.losses import avg_std as _avg_std
                from mini_vjepa.losses import avg_cosine_sim as _avg_cos
                from mini_vjepa.losses import effective_rank as _eff_rank
                avg_std_v = float(_avg_std(last_z_pred_detached).item())
                eff_rank_v = float(_eff_rank(last_z_pred_detached).item())
                cos_sim_v = float(_avg_cos(last_z_pred_detached).item())
                # Encoder-level monitors: if the encoder is what collapses
                # (as H2 predicts), these should drop before z_pred does.
                if last_ctx_detached is not None:
                    ctx_std_v = float(_avg_std(last_ctx_detached).item())
                    ctx_rank_v = float(_eff_rank(last_ctx_detached).item())
                    ctx_cos_v = float(_avg_cos(last_ctx_detached).item())
                else:
                    ctx_std_v = 0.0
                    ctx_rank_v = 0.0
                    ctx_cos_v = 0.0

        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "tau": tau,
            "loss": avg_loss,
            "mse": avg_mse,
            "var_reg": avg_var,
            "cov_reg": avg_cov,
            "avg_std": avg_std_v,
            "effective_rank": eff_rank_v,
            "avg_cosine_sim": cos_sim_v,
            "ctx_avg_std": ctx_std_v,
            "ctx_effective_rank": ctx_rank_v,
            "ctx_avg_cosine_sim": ctx_cos_v,
            "time_s": elapsed,
        }
        metrics_history.append(row)
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([row[k] for k in metrics_columns])
        print(
            f"[epoch {epoch:03d}] loss={avg_loss:.4f} mse={avg_mse:.4f} var={avg_var:.4f} "
            f"cov={avg_cov:.4f} avg_std={avg_std_v:.4f} eff_rank={eff_rank_v:.2f} "
            f"cos_sim={cos_sim_v:.4f} ctx_eff_rank={ctx_rank_v:.2f} ctx_cos={ctx_cos_v:.4f} "
            f"lr={current_lr:.2e} tau={tau:.5f} t={elapsed:.1f}s",
            flush=True,
        )
        avg_epoch_s = sum(r["time_s"] for r in metrics_history) / len(metrics_history)
        remaining = (epochs - epoch - 1) * avg_epoch_s
        print(
            f"[ETA] avg={avg_epoch_s:.1f}s/epoch, {epochs - epoch - 1} epochs left, "
            f"~{remaining/60:.1f} min remaining (~{remaining/3600:.2f} h)",
            flush=True,
        )

        # Anti-collapse warning: avg_std catches per-dim variance collapse.
        if avg_std_v < 0.05:
            low_std_streak += 1
        else:
            low_std_streak = 0
        if low_std_streak >= 5:
            print(
                f"[anti-collapse WARNING] avg_std<0.05 for {low_std_streak} consecutive epochs",
                file=sys.stderr, flush=True,
            )

        # Dimensional-collapse guard: effective_rank catches the case where
        # per-dim std looks fine but every dim encodes the same signal.
        # NaN counts as collapsed.
        #
        # Trigger on EITHER z_pred OR encoder-output (ctx) effective_rank
        # below threshold. The ctx check is load-bearing: there is a failure
        # mode where z_pred eff_rank looks fine (20+) while the encoder is
        # fully collapsed (ctx_eff_rank ~ 3), because the predictor's own
        # pos_embed + mask_token can fake diversity from its own parameters.
        z_collapsed = (eff_rank_v != eff_rank_v) or (eff_rank_v < collapse_eff_rank_threshold)
        ctx_collapsed = (ctx_rank_v != ctx_rank_v) or (ctx_rank_v < collapse_eff_rank_threshold)
        is_collapsed = z_collapsed or ctx_collapsed
        if is_collapsed:
            collapse_streak += 1
        else:
            collapse_streak = 0
        if collapse_streak >= 5:
            which = "ctx_eff_rank" if ctx_collapsed and not z_collapsed else (
                "eff_rank" if z_collapsed and not ctx_collapsed else "both"
            )
            print(
                f"[anti-collapse WARNING] {which}<{collapse_eff_rank_threshold} "
                f"for {collapse_streak} consecutive epochs (dimensional collapse)",
                file=sys.stderr, flush=True,
            )
        if collapse_streak >= collapse_patience:
            print(
                f"[anti-collapse STOP] eff_rank<{collapse_eff_rank_threshold} for "
                f"{collapse_streak} epochs >= patience {collapse_patience}. Aborting at epoch {epoch}.",
                file=sys.stderr, flush=True,
            )
            break

        # Checkpointing
        is_periodic = ((epoch + 1) % checkpoint_every == 0) or (epoch + 1 == epochs)
        if is_periodic:
            ckpt_path = run_dir / f"ckpt_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "ema_tau": tau,
                "metrics_history": metrics_history,
            }, ckpt_path)
            saved_ckpts.append(ckpt_path)
            # FIFO keep at most max_checkpoints (excluding the best)
            while len(saved_ckpts) > max_checkpoints:
                oldest = saved_ckpts.pop(0)
                if oldest != best_ckpt_path and oldest.exists():
                    oldest.unlink()

        # Best by effective_rank, but only after warmup and only if not collapsed.
        # Previously the random init (epoch 0) won by default when training
        # subsequently collapsed - that selected an untrained model.
        warmup_epochs_int = int(cfg["training"]["warmup_epochs"])
        eligible_for_best = (
            epoch >= warmup_epochs_int
            and eff_rank_v == eff_rank_v  # not NaN
            and eff_rank_v >= collapse_eff_rank_threshold
        )
        if eligible_for_best and eff_rank_v > best_eff_rank:
            best_eff_rank = eff_rank_v
            new_best = run_dir / "ckpt_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "ema_tau": tau,
                "metrics_history": metrics_history,
                "best_effective_rank": best_eff_rank,
            }, new_best)
            best_ckpt_path = new_best

        # MPS cache hygiene
        if device.type == "mps" and (epoch + 1) % 20 == 0:
            torch.mps.empty_cache()

    print(f"[train] done. run_dir={run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
