"""
main.py — Command-line entry point for the Handwriting Personality AI.

Usage
-----
  python main.py --image path/to/handwriting.jpg
  python main.py --image path/to/handwriting.jpg --no-deep
  python main.py --image path/to/handwriting.jpg --no-save
"""

import argparse
import sys
import os

# Make sure src/ is importable regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from src.main_pipeline import run_pipeline
from src.utils.helper  import logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse handwriting and predict Big-Five personality traits."
    )
    p.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the handwriting image (JPG / PNG / BMP).",
    )
    p.add_argument(
        "--no-deep",
        action="store_true",
        default=False,
        help="Skip ResNet + ViT deep feature extraction (faster, less accurate).",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Do not save results to disk (useful for quick checks).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        logger.error("Image not found: %s", args.image)
        sys.exit(1)

    result = run_pipeline(
        image_path        = args.image,
        use_deep_features = not args.no_deep,
        save_outputs      = not args.no_save,
    )

    # ── Print personality summary to console ──────────────────────────────
    scores  = result["personality"]["scores"]
    labels  = result["personality"]["labels"]
    method  = result["personality"]["method"]

    print("\n" + "═" * 62)
    print("  HANDWRITING PERSONALITY ANALYSIS")
    print("═" * 62)
    print(f"  Image  : {os.path.basename(args.image)}")
    print(f"  Method : {method}")
    print(f"  Time   : {result['elapsed_sec']}s")
    print()

    for trait, score in scores.items():
        bar = "█" * int(score * 25) + "░" * (25 - int(score * 25))
        print(f"  {trait:<18} {bar}  {score:.2f}")
        print(f"  {'':18} → {labels[trait]}")
        print()

    # ── Print saved file paths ────────────────────────────────────────────
    paths = result.get("output_paths", {})
    if paths:
        print("─" * 62)
        print("  Saved outputs:")
        for name, path in paths.items():
            print(f"    {name:<20} {path}")

    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()
