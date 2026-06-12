"""Create comparison plots from saved training metrics."""

from __future__ import annotations

import argparse

from snake_ai.training.metrics import load_metrics, plot_length_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Snake training metrics.")
    parser.add_argument("metrics", nargs="+", help="Metrics JSON files to compare.")
    parser.add_argument("--output", default="outputs/training_comparison.png")
    parser.add_argument(
        "--window",
        type=int,
        default=500,
        help="Rolling-average window in episodes (default: 500).",
    )
    args = parser.parse_args()
    datasets = [load_metrics(path) for path in args.metrics]
    output = plot_length_comparison(datasets, args.output, window=args.window)
    print(f"Comparison plot saved to {output}")


if __name__ == "__main__":
    main()
