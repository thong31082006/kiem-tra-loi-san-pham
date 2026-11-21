"""Re-evaluate the active HOG + SVM on full Test images, without retraining."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.hog_svm import ARTIFACT_ROOT, HogSvmService
from ml.train import DEFAULT_MANIFEST
from scripts.evaluate_web_pipeline import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evaluate(args.manifest, args.artifact_root, output=args.output, service=HogSvmService(args.artifact_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
