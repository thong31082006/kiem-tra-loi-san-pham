"""Download the public PET bottle image dataset from Kaggle, without model weights."""
import argparse
import hashlib
import json
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

APP_DIR = Path(__file__).resolve().parents[1]
DATASET_ID = "dhiaalhemdani/industrial-inspection-system"
VERSION = 10
SOURCE_URL = f"https://www.kaggle.com/datasets/{DATASET_ID}"
DOWNLOAD_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_ID}"
    f"?datasetVersionNumber={VERSION}"
)
DESTINATION = APP_DIR / "data" / "datasets" / "plastic_bottles"
ALLOWED_ROOTS = {"images", "labels", "labels_seg", "annotations_labelme"}
ALLOWED_FILES = {"README.md", "LICENSE", "LICENSE.txt", "data_detect.yaml", "data_segment.yaml"}
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
EXPECTED_ARCHIVE_SHA256 = "d2f580abf8b210a0b8a0b23c4de1edda974abb95e96b1c087631e7e812ab1f93"


def download_archive(target):
    request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "BottleInspection/1.0"})
    digest = hashlib.sha256()
    received = 0
    next_report = 20 * 1024 * 1024
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as output:
        while block := response.read(1024 * 1024):
            received += len(block)
            if received > MAX_ARCHIVE_BYTES:
                raise ValueError("The Kaggle archive is unexpectedly large.")
            output.write(block)
            digest.update(block)
            if received >= next_report:
                print(f"Downloaded {received // (1024 * 1024)} MiB", flush=True)
                next_report += 20 * 1024 * 1024
    if not zipfile.is_zipfile(target):
        raise ValueError("Kaggle did not return a ZIP archive.")
    return digest.hexdigest(), received


def extract_dataset(archive_path, destination, allowed_roots=None, allowed_files=None):
    destination = destination.resolve()
    allowed_roots = ALLOWED_ROOTS if allowed_roots is None else allowed_roots
    allowed_files = ALLOWED_FILES if allowed_files is None else allowed_files
    extracted = []
    with zipfile.ZipFile(archive_path) as archive:
        planned = []
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if member.is_dir():
                continue
            if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Symbolic links are not allowed: {member.filename}")
            if name.parts[0] not in allowed_roots and member.filename not in allowed_files:
                continue
            output = (destination / Path(*name.parts)).resolve()
            output.relative_to(destination)
            exists = output.exists()
            if exists:
                identical = output.is_file() and output.stat().st_size == member.file_size
                if identical:
                    checksum = 0
                    with output.open("rb") as existing:
                        while block := existing.read(1024 * 1024):
                            checksum = zlib.crc32(block, checksum)
                    identical = checksum == member.CRC
                if not identical:
                    raise ValueError(
                        f"Existing file differs from Kaggle and was preserved: {output}. "
                        "Move your edited file elsewhere before downloading again."
                    )
            planned.append((member, output, exists))
        # Validate every existing source file before creating any new files.
        for member, output, exists in planned:
            if exists:
                extracted.append(member.filename)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            created = False
            try:
                with archive.open(member) as source, output.open("xb") as target:
                    created = True
                    shutil.copyfileobj(source, target)
            except Exception:
                if created:
                    output.unlink(missing_ok=True)
                raise
            extracted.append(member.filename)
    return extracted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded Kaggle ZIP.")
    args = parser.parse_args()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    downloaded_here = args.archive is None
    if downloaded_here:
        with tempfile.NamedTemporaryFile(prefix=".download-", suffix=".zip", dir=DESTINATION, delete=False) as temporary:
            archive_path = Path(temporary.name)
    else:
        archive_path = args.archive.resolve()
    try:
        if downloaded_here:
            print(f"Downloading {DATASET_ID}, version {VERSION}", flush=True)
            archive_hash, archive_size = download_archive(archive_path)
        else:
            with archive_path.open("rb") as source:
                archive_hash = hashlib.file_digest(source, "sha256").hexdigest()
            archive_size = archive_path.stat().st_size
        if archive_hash != EXPECTED_ARCHIVE_SHA256:
            raise ValueError("The archive checksum does not match the verified Kaggle version 10.")
        files = extract_dataset(archive_path, DESTINATION)
        image_files = [name for name in files if name.startswith("images/") and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not image_files:
            raise ValueError("The archive did not contain bottle images.")
        metadata = {
            "dataset_id": DATASET_ID,
            "version": VERSION,
            "source_url": SOURCE_URL,
            "license": "MIT (as declared on the Kaggle data card)",
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_sha256": archive_hash,
            "archive_bytes": archive_size,
            "image_count": len(image_files),
            "extracted_file_count": len(files),
            "excluded": "benchmarks, model weights, and other non-dataset files",
            "files": files,
        }
        (DESTINATION / "source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Ready: {len(image_files)} images, {len(files)} extracted files at {DESTINATION}", flush=True)
    finally:
        if downloaded_here:
            archive_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Dataset download failed: {error}", file=sys.stderr)
        sys.exit(1)
