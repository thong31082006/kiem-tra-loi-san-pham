"""Cache pixel-only bottle detections so training uses the same crop as the website."""
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from ml.features import APP_DIR, read_bgr
from ml.localization import BottleLocator

MANIFEST = APP_DIR / "data" / "training" / "manifest.json"
INDEX = MANIFEST.parent / "crop_index.json"
CACHE_DIR = MANIFEST.parent / "crops" / "ssdlite_v1"


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    started = time.perf_counter()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    previous = json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.is_file() else {}
    records = manifest["records"]
    cached, retained, rejected = {}, [], []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    locator = BottleLocator()
    for position, record in enumerate(records):
        destination = CACHE_DIR / f"{record['id']}.png"
        entry = previous.get(record["id"])
        if entry and entry.get("source_sha256") == record.get("sha256") and destination.is_file():
            detection = entry["detection"]
        else:
            try:
                cropped, detection = locator.locate(read_bgr(record["path"]))
            except ValueError as error:
                if not str(error).startswith(("Không tìm thấy chai", "Ảnh có nhiều chai")):
                    raise
                rejected.append({"id": record["id"], "path": record["path"], "label": record["label"], "split": record["split"], "reason": str(error)})
                continue
            ok, encoded = cv2.imencode(".png", cropped)
            if not ok:
                raise ValueError(f"Cannot encode crop for {record['id']}")
            encoded.tofile(str(destination))
        cached[record["id"]] = {"source_sha256": record.get("sha256"), "detection": detection}
        retained.append({**record, "training_path": destination.relative_to(APP_DIR).as_posix(), "auto_detection": detection})
        if (position + 1) % 40 == 0 or position + 1 == len(records):
            atomic_json(INDEX, cached)
            print(f"Automatic bottle crops {position + 1}/{len(records)}; rejected={len(rejected)}", flush=True)
    counts = {split: {**dict(Counter(row["label"] for row in retained if row["split"] == split)),
                      "total": sum(row["split"] == split for row in retained)} for split in ("train", "val", "test")}
    for split, labels in counts.items():
        if min(labels.get("normal", 0), labels.get("defect", 0)) < 10:
            raise ValueError(f"Insufficient class coverage after automatic cropping: {split} {labels}")
    manifest.update({"version": 3, "input_mode": "auto_bottle_roi", "records": retained,
                     "image_count": len(retained), "counts": counts,
                     "crop_preprocessing": {"detector": "SSDLite320 MobileNetV3 Large COCO_V1",
                         "device": "cpu", "score_threshold": 0.30, "padding": 0.08,
                         "label_independent": True, "cache_format": "lossless PNG",
                         "created_at": datetime.now(timezone.utc).isoformat(),
                         "seconds": round(time.perf_counter() - started, 2),
                         "rejected_count": len(rejected), "rejected": rejected}})
    atomic_json(INDEX, cached)
    atomic_json(MANIFEST, manifest)
    print(json.dumps({"used": len(retained), "counts": counts, "rejected": len(rejected)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
