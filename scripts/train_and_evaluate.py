"""Web job: fit all three models, then evaluate the complete pixel-only pipeline."""
import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.features import APP_DIR
from ml.train import DEFAULT_ARTIFACT_ROOT, _atomic_json, train_all
from scripts.evaluate_web_pipeline import evaluate
from scripts.export_report import export_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=APP_DIR / "data/training/manifest.json")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    progress_path = args.artifact_root / "training_progress.json"
    try:
        train_all(args.manifest, args.epochs, args.batch_size, args.device, args.seed, args.artifact_root)
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress.update({"state": "running", "running": True, "model": "evaluation", "message": "Đã huấn luyện. Đang đánh giá toàn bộ luồng web trên ảnh Test."})
        _atomic_json(progress_path, progress)
        evaluate(args.manifest, args.artifact_root)
        export_report(args.artifact_root, None if args.artifact_root.resolve() == DEFAULT_ARTIFACT_ROOT.resolve() else args.artifact_root / "summary.md")
        progress.update({"state": "complete", "running": False, "message": "Đã huấn luyện ba mô hình và đánh giá toàn bộ luồng web."})
        _atomic_json(progress_path, progress)
    except Exception as error:
        traceback.print_exc()
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
        progress.update({"state": "failed", "running": False, "message": f"Tác vụ dừng: {error}"})
        _atomic_json(progress_path, progress)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
