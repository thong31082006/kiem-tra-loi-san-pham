"""Download the annotated Ron 88 bottle defect dataset from public Kaggle."""
import hashlib
import json
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from download_dataset import extract_dataset

APP_DIR = Path(__file__).resolve().parents[1]
DATASET_ID = "jihadakbr/water-bottle-defect-level-detection-dataset"
VERSION = 1
SOURCE_URL = f"https://www.kaggle.com/datasets/{DATASET_ID}"
DOWNLOAD_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_ID}"
    f"?datasetVersionNumber={VERSION}"
)
DESTINATION = APP_DIR / "data" / "datasets" / "bottle_defects"
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
EXPECTED_ARCHIVE_SHA256 = "db30051da47788f0caeede7d9c56100822af834920db9a9ad2bdd5a263730666"


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".download-", suffix=".zip", dir=DESTINATION, delete=False) as temporary:
        archive_path = Path(temporary.name)
    try:
        digest = hashlib.sha256()
        received = 0
        next_report = 40 * 1024 * 1024
        print(f"Downloading {DATASET_ID}, version {VERSION}", flush=True)
        request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "BottleInspection/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as output:
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > MAX_ARCHIVE_BYTES:
                    raise ValueError("The Kaggle archive is unexpectedly large.")
                output.write(block)
                digest.update(block)
                if received >= next_report:
                    print(f"Downloaded {received // (1024 * 1024)} MiB", flush=True)
                    next_report += 40 * 1024 * 1024
        archive_hash = digest.hexdigest()
        if EXPECTED_ARCHIVE_SHA256 and archive_hash != EXPECTED_ARCHIVE_SHA256:
            raise ValueError("Archive checksum does not match the verified Kaggle version 1.")
        if not zipfile.is_zipfile(archive_path):
            raise ValueError("Kaggle did not return a ZIP archive.")
        files = extract_dataset(archive_path, DESTINATION, allowed_roots={"images", "labels"}, allowed_files={"data.yaml"})
        image_count = sum(name.startswith("images/") and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"} for name in files)
        label_count = sum(name.startswith("labels/") and name.endswith(".txt") for name in files)
        if image_count != 1500 or label_count != 1500:
            raise ValueError(f"Unexpected source counts: {image_count} images and {label_count} labels.")
        metadata = {
            "dataset_id": DATASET_ID,
            "version": VERSION,
            "source_url": SOURCE_URL,
            "author_repository": "https://github.com/jihadakbr/water-bottle-quality-inspection",
            "license": "MIT (as declared on the Kaggle data card)",
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_sha256": archive_hash,
            "archive_bytes": received,
            "image_count": image_count,
            "label_count": label_count,
            "files": files,
        }
        (DESTINATION / "source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Ready: {image_count} images, {label_count} annotations at {DESTINATION}", flush=True)
        print(f"Archive SHA256: {archive_hash}", flush=True)
    finally:
        archive_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Dataset download failed: {error}", file=sys.stderr)
        sys.exit(1)
