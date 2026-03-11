"""
Run the full training pipeline in one command.

Usage:
    python run_pipeline.py

Executes in order:
    1. Build training dataset  (features/build_dataset.py)
    2. Train all models        (models/train.py)
    3. Evaluate on test set    (models/evaluate.py)
"""

import os
import sys
import time

# Make all project packages importable
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _banner(step_num: int, total: int, title: str):
    print()
    print("=" * 62)
    print(f"  STEP {step_num}/{total} — {title}")
    print("=" * 62)
    print()


def main():
    start_total = time.time()

    # ------------------------------------------------------------------
    # Step 1 — Build training dataset
    # ------------------------------------------------------------------
    _banner(1, 3, "BUILD TRAINING DATASET")
    from features.build_dataset import build_training_dataset

    t0 = time.time()
    dataset = build_training_dataset(save=True)
    elapsed = time.time() - t0

    if dataset.empty:
        print("\n  PIPELINE ABORTED: Dataset build failed.")
        return
    print(f"\n  Step 1 complete. ({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # Step 2 — Train all models
    # ------------------------------------------------------------------
    _banner(2, 3, "TRAIN ALL MODELS")
    from models.train import train_all

    t0 = time.time()
    results = train_all()
    elapsed = time.time() - t0
    print(f"\n  Step 2 complete. ({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # Step 3 — Evaluate models
    # ------------------------------------------------------------------
    _banner(3, 3, "EVALUATE MODELS")
    from models.evaluate import evaluate_all

    t0 = time.time()
    evaluate_all()
    elapsed = time.time() - t0
    print(f"\n  Step 3 complete. ({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    total_elapsed = time.time() - start_total
    print()
    print("=" * 62)
    print(f"  PIPELINE COMPLETE — Total time: {total_elapsed:.1f}s")
    print()
    print("  Next steps:")
    print("    python app/main.py AAPL    ← run a prediction")
    print("    python app/main.py NVDA")
    print("=" * 62)


if __name__ == "__main__":
    main()
