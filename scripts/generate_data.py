"""CLI to generate a billiard dataset from a YAML config."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.generator import generate_dataset, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate billiard dataset.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--n-sequences", type=int, default=None,
                        help="Override n_sequences from the config (useful for sample data).")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output path from the config.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    cfg, _ = load_config(args.config)
    if args.n_sequences is not None:
        cfg.n_sequences = int(args.n_sequences)
    if args.output is not None:
        cfg.output_path = args.output
    if args.seed is not None:
        cfg.seed = int(args.seed)

    out = generate_dataset(cfg, show_progress=not args.no_progress)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
