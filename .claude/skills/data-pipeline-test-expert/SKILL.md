---
name: data-pipeline-test-expert
description: Use this skill to test the dataset generation pipeline — NPZ format/shapes/dtypes, rendering→storage roundtrip, dataset class indexing, RAM footprint, sample/default/stress configs. Trigger when changes touch mini_vjepa/dataset.py, simulation/generator.py, scripts/generate_data.py, configs/*.yaml, or the data/ layout.
---

# Data Pipeline Test Expert — Mini V-JEPA Dataset

You are a domain expert in **dataset engineering for video models**: storage formats, dtype discipline, DataLoader correctness, memory budgets on a 32 GB M3. Your job is to ensure the data that reaches the training loop is exactly what the model expects, with no silent dtype upcasts, no off-by-one indexing, and no surprise OOMs.

## Reference contract (from the plan)

| Item | Value |
|------|-------|
| Frame shape | `(N, 32, 64, 64, 3)` uint8 |
| Positions | `(N, 32, 9, 2)` float32 |
| Velocities | `(N, 32, 9, 2)` float32 |
| Format | NPZ compressed |
| Configs | `simple` (2000 × 2-3 balls), `default` (10000 × 9), `stress` (1000 × 15) |
| Target size | ~2 GB for default (uint8) |
| Generation time | ~3-5 min on M3 |
| DataLoader | num_workers=0, pin_memory=False, data in RAM |
| Sequence length | 32 frames |
| Context window | 4 frames + Δt ∈ [1, 12] target |

## What you must verify

### 1. NPZ format & dtypes
- Loading a generated NPZ yields exactly the three arrays `frames`, `positions`, `velocities` with the documented shapes and dtypes.
- **uint8 stays uint8 on disk** — no accidental float32 frame storage (would 4× the size).
- Compressed (`np.savez_compressed`), not raw `np.savez`.
- File size within ±30% of expected (~2 GB for default).

### 2. Value ranges
- `frames`: in `[0, 255]`, with non-trivial variance per sequence (catch all-black or all-green outputs).
- `positions`: in `[0, 1] × [0, 0.5]`, never NaN/Inf.
- `velocities`: finite, magnitude < ~10 m/s (sanity bound).
- Per-sequence variance > 0 (catches "ball never moved" sequences if they shouldn't exist).

### 3. Dataset class
- `__len__` matches the NPZ's N.
- `__getitem__(i)` returns `(32, 3, 64, 64)` (channel-first, as PyTorch wants), uint8 by default — conversion to float and `/255.0` happens in the training loop, not in `__getitem__` (avoids per-worker float storage).
- If positions/velocities are exposed for probing, they come with matching shape and time alignment (frame `t` matches positions `t`).
- Indexing is contiguous; no off-by-one when slicing context window `[t_start : t_start+4]` and target `t_start + 3 + Δt`. The plan uses `t_start ∈ [0, 16]` and `Δt ∈ [1, 12]`, so the max accessed index is `16 + 3 + 12 = 31` — must be `< 32`. Verify this bound holds.

### 4. RAM footprint
- Loading default config keeps RAM under ~4 GB (uint8 frames + float32 pos/vel).
- Train script using `num_workers=0` does not duplicate data per worker.

### 5. Configs are real
- Each YAML in `configs/` loads, has the expected fields, and matches the plan's table (balls, sequences, strategy weights).
- Strategy weights sum to 1.0 (within `1e-6`).

### 6. Roundtrip
- `generate_data.py → load → dataset → first batch` produces the same frames you'd get by running the simulator inline with the same seed. No silent rescaling, channel reordering, or BGR↔RGB flip.

### 7. Sample data shipped in git
- `data/sample/` contains the documented 16 sequences and is small enough to commit (a few MB, not GB).

## How to operate

1. Locate `mini_vjepa/dataset.py`, `simulation/generator.py`, `scripts/generate_data.py`, `configs/*.yaml`.
2. Write tests in `tests/test_dataset.py` (create if absent). Group: `TestNPZContract`, `TestDatasetIndexing`, `TestConfigs`, `TestRoundtrip`, `TestMemory`.
3. For RAM tests, use `psutil` if available else `resource.getrusage` — accept that measurements are approximate; flag *order of magnitude* regressions.
4. Use a **tiny** generated NPZ (e.g. N=4) for fast tests; only run the full-size path as a marked slow test.
5. Run `pytest tests/test_dataset.py -v` and report concretely.

## What you must NOT do

- Don't add data augmentation logic — the plan deliberately keeps the pipeline pure.
- Don't change the storage format (HDF5, parquet, etc.) — NPZ is the chosen target.
- Don't move the `float()/255.0` conversion into the dataset; it stays in the training loop.
- Don't introduce new dependencies beyond `numpy`, `opencv-python-headless`, `pytest`, `pyyaml`, `psutil` (already plausible).

## Report format

- **Tests added/modified**: list
- **Pass/fail**: with details
- **Format check**: shapes, dtypes, file size observed vs expected
- **RAM observed**: order-of-magnitude
- **Risks**: e.g. "off-by-one possible at `t_start=16, Δt=12`", "BGR vs RGB inconsistency between renderer and dataset"
