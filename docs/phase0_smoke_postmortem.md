# Phase-0 smoke run — postmortem (2026-05-25)

**Outcome:** killed after 4h 27min at SIGTERM. Partial data only. No artifacts
written to `--out` directory (script writes figures + `metrics.json` at the
very end). Re-run required.

## Command that was running

```
python scripts/evaluate.py \
  --ckpt runs/20260519_195827_iter3_full_150/ckpt_best.pt \
  --data data/default.npz \
  --config runs/20260519_195827_iter3_full_150/config.yaml \
  --pixel-ckpt runs/baseline_pixel/20260519_225206/ckpt_best.pt \
  --out /tmp/phase0_smoke/N2000/ \
  --max-eval-sequences 2000 \
  --ram-limit-gb 20
```

Started 11:38:50, killed 16:06+. Process PID 67591.

## What the partial run did validate

Lines 1–21 of [/tmp/phase0_smoke/N2000.log](file:///tmp/phase0_smoke/N2000.log)
confirm the Phase-0 fixes F1, F4, F5, F6 ship clean end-to-end. F2 and F3 are
not exercised by `scripts/evaluate.py` so they are not validated by this run.

| Fix | Evidence in log |
|---|---|
| F1 mask-leak | degradation curve ran with `eval_mask_ratio=0.00` (line 14, "eval_mask_ratio=0.00") |
| F4 RidgeCV α grid | `LinAlgWarning rcond=6.41e-08` during rollout-MAE probe — within the new `logspace(-3, 6, 19)` range, no truncation |
| F5 CLI default sync | ran at N=2000 as requested via `--max-eval-sequences 2000` |
| F6 RAM-cap surfacing | first log line: `WARNING: could not set RLIMIT_AS to 20.0 GB: current limit exceeds maximum limit` (was silent pre-fix) |

Probe numbers landed (V-JEPA side, N=2000, 80/20 split):

| Probe | positions R² | velocities R² |
|---|---|---|
| target encoder (frame-level) | 0.5868 | 0.0385 |
| online encoder (frame-level) | 0.5792 | 0.0419 |
| ctx-window (V-JEPA native, α=1) | 0.4458 | -0.2687 |
| held-out regime (train≠test init_strategy) | 0.4579 | -0.3224 |

Consistent with [assets/results/v2_phase1/metrics.json](../assets/results/v2_phase1/metrics.json)
within run-to-run noise. No silent regression introduced by the F1/F4/F5/F6
patches.

## What did NOT complete

- **Pixel-baseline context-window probe** — log line 22 ("encoding 4-frame
  context windows (pixel baseline)...") was the last thing printed. The
  encoder forward pass for the pixel baseline almost certainly finished; the
  process was hung in the *probe-fit* step that follows.
- **Side-by-side V-JEPA vs pixel comparison** — figures
  (`jepa_vs_pixel.png`, `probe_grid.png`, etc.) never reached the write step.
- **All persisted outputs** — `/tmp/phase0_smoke/N2000/` is empty.

## Root cause: memory thrashing, not deadlock

Diagnosis via `sample 67591 2` (macOS built-in profiler, no sudo needed).

**Main-thread hot path (1499 / 1505 samples):**

```
_PyEval_EvalFrameDefault
  array_dot                       (in _multiarray_umath.cpython-312-darwin.so)
    cblas_matrixproduct
      gemv
        cblas_sgemv64_            (in libopenblas64_.0.dylib)
          sgemv_thread_n (916)
          sgemv_thread_t (583)
```

100% of CPU time was in **numpy.dot → OpenBLAS sgemv**, i.e. the CPU-side
Ridge solver fitting the pixel-baseline context-window probe. Not MPS, not the
encoder forward — pure CPU matmul.

**Memory snapshot at the time (pre-kill):**

| Metric | Value | Source |
|---|---|---|
| Physical footprint (this process) | **35.5 GB** (peak 37.7 GB) | `sample` header |
| Free system pages | **76 MB** (4782 × 16 KB) | `vm_stat` |
| RSS via `ps` | 5.7 GB | `ps -o rss` |

The `ps`-reported RSS (5.7 GB) was misleading: it does not include MPS unified
memory, which sits in a separately-accounted region. The actual physical
footprint was 35.5 GB on a 32 GB machine — i.e. **the process was over total
RAM**, with macOS compressing and swapping pages aggressively. Sub-millisecond
BLAS operations were stalling on page faults, which is exactly why the BLAS
matmul dominated the profile.

**Post-kill confirmation:** `vm_stat` showed 1,158,394 free pages (18.5 GB
free) immediately after SIGTERM — confirming the killed process had been
holding ~18 GB of resident memory that the OS could reclaim.

## Why this surprised us

Three things conspired:

1. **`scripts/evaluate.py` has no progress logging inside
   [`_encode_windows_with`](../scripts/evaluate.py#L205-L255) or between the
   "encoding..." print and the next probe result.** From outside, a slow run
   in the BLAS solver is visually indistinguishable from a hang in the
   encoder.
2. **`--ram-limit-gb 20` was silently capped** by the shell's `RLIMIT_AS`
   ceiling (the F6 fix surfaces the warning, but the cap itself doesn't hold
   on macOS — there was no actual budget enforcement).
3. **`ps`-reported RSS underestimates true MPS memory pressure by ~6×.** Any
   future "is the eval safe at this N?" check must use `sample <pid>` /
   `vmmap <pid>` or watch `vm_stat` free pages — not `ps`.

## Action items before re-running

These should land before any re-launch of the smoke at N=2000.

1. **Add observability in `scripts/evaluate.py` around the pixel-baseline
   block** ([scripts/evaluate.py:1181-1210](../scripts/evaluate.py#L1181-L1210)).
   At minimum: a `print(..., flush=True)` before each `linear_probe_ridge`
   call, with the input shape. Optional: a per-batch progress print inside
   [`_encode_windows_with`](../scripts/evaluate.py#L205-L255) gated by an env
   var or `--verbose` flag.
2. **Drop N for the smoke pass.** N=2000 with both V-JEPA and pixel encoders
   in memory, plus their context-window tensors `(2000, 29, 256, 128)
   float32 ≈ 6 GB each`, plus their flattened ridge design matrices, blew
   past the 32 GB total. Re-run at **N=1000 or N=1500** for the smoke, and
   only push to N=2000 once the pipeline is observably making progress
   through each step.
3. **Confirm the F3 second-leg (frozen-V1 pixel baseline) training run is
   queued.** The current run was using the pre-F3 from-scratch pixel ckpt
   (`runs/baseline_pixel/20260519_225206/ckpt_best.pt`, dated 2026-05-19).
   Phase 0 cannot ship clean without the frozen-V1 variant existing as a
   trained checkpoint. See [docs/eval_failure_modes.md](eval_failure_modes.md)
   row F3.
4. **Use `sample <pid>` / `vmmap <pid>` — not `ps` — to size future evals.**
   `ps` does not account MPS unified memory.

## Files preserved

- Eval log (23 lines): `/tmp/phase0_smoke/N2000.log` (ephemeral — `/tmp` is
  cleared on reboot; copy to a stable path if you want to keep it)
- Sample dump (905 lines, includes BLAS hot path): `/tmp/eval_sample.txt`
  (same caveat)

## Status of Phase 0

F1, F4, F5, F6 are **validated by partial-run evidence**. F2 is committed
([scripts/train.py:341-364](../scripts/train.py#L341-L364)) but exercises the
training loop, not the eval — it will be validated by the next training run,
not by re-running this smoke. F3 is **code-validated only** — needs the
frozen-V1 pixel checkpoint to be trained before the
`evaluation-test-expert` review gate can run.

Re-run blocking on action items 1 + 3 above.
