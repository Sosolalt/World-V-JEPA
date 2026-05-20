"""Compare metrics.csv from multiple runs and print a side-by-side summary.

Usage:
    python scripts/compare_runs.py runs/*E0_def* runs/*E1_def*

For each run, prints a compact table of (epoch, eff_rank, cos_sim, loss) at a
few diagnostic checkpoints (0, 10, 20, 28, last) plus the min eff_rank reached
and a 'verdict' (HEALTHY | DEGRADED | COLLAPSED).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


CHECKPOINTS = [0, 10, 15, 20, 25, 28, 29]


def load_metrics(run_dir: Path) -> list[dict]:
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return [{k: v for k, v in row.items()} for row in csv.DictReader(f)]


def verdict(rows: list[dict]) -> str:
    if not rows:
        return "MISSING"
    ranks = [float(r["effective_rank"]) for r in rows if r["effective_rank"] and r["effective_rank"] != "nan"]
    ctx_ranks = []
    for r in rows:
        v = r.get("ctx_effective_rank", "")
        if v and v != "nan":
            try:
                ctx_ranks.append(float(v))
            except ValueError:
                pass
    if not ranks:
        return "MISSING"
    post_warmup = ranks[10:] if len(ranks) > 10 else ranks
    if any(v != v for v in post_warmup):  # NaN check
        return "NaN-CRASH"
    min_post_warmup_z = min(post_warmup)
    min_post_warmup_ctx = min(ctx_ranks[10:]) if len(ctx_ranks) > 10 else (min(ctx_ranks) if ctx_ranks else 999)
    if min_post_warmup_z < 5.0 or min_post_warmup_ctx < 5.0:
        return "COLLAPSED"
    if min_post_warmup_z < 10.0 or min_post_warmup_ctx < 10.0:
        return "DEGRADED"
    return "HEALTHY"


def fmt_row(rows: list[dict], epoch: int) -> str:
    if not rows or epoch >= len(rows):
        return "        —"
    r = rows[epoch]
    er = float(r["effective_rank"]) if r["effective_rank"] else float("nan")
    cer = float(r.get("ctx_effective_rank", "0") or "0") if r.get("ctx_effective_rank") else float("nan")
    # Use 'er/cer' so both encoder + predictor rank visible at a glance.
    return f"{er:5.1f}/{cer:5.1f}"


def main(args: list[str]) -> int:
    if not args:
        print(__doc__)
        return 1
    runs = sorted({Path(a) for a in args if Path(a).is_dir()})
    if not runs:
        print("no run dirs found", file=sys.stderr)
        return 2

    name_w = max(len(r.name) for r in runs)
    header = f"{'run (z_er/ctx_er)':<{name_w}}  " + "  ".join(f"ep{e:02d}".center(12) for e in CHECKPOINTS) + "  verdict"
    print(header)
    print("-" * len(header))
    for run in runs:
        rows = load_metrics(run)
        cells = [fmt_row(rows, e).center(12) for e in CHECKPOINTS]
        v = verdict(rows)
        ranks = [float(r["effective_rank"]) for r in rows if r["effective_rank"] and r["effective_rank"] != "nan"]
        ctx_ranks_all = []
        for r in rows:
            cv = r.get("ctx_effective_rank", "")
            if cv and cv != "nan":
                try:
                    ctx_ranks_all.append(float(cv))
                except ValueError:
                    pass
        min_rank = min(ranks) if ranks else float("nan")
        min_ctx = min(ctx_ranks_all) if ctx_ranks_all else float("nan")
        print(f"{run.name:<{name_w}}  " + "  ".join(cells) + f"  {v} (min z={min_rank:.1f}, ctx={min_ctx:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
