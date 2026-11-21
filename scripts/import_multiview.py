"""Validate real multi-view captures and stage them for review, never auto-train."""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from ml.features import APP_DIR, read_bgr
from ml.train import _atomic_json

VIEWS = {"front", "back", "left", "right", "top", "bottom", "tilted", "horizontal", "upside_down"}
TYPES = {"defect_low_fill", "defect_no_cap", "defect_loose_cap", "defect_debris", "defect_label_damage"}
REQUIRED = {"path", "bottle_id", "session_id", "view", "defect_type", "defect_visible"}
DESTINATION = APP_DIR / "data" / "datasets" / "multiview"


def visible_label(defect_type, visibility):
    types = set(defect_type.split(";"))
    if visibility not in {"true", "false"}:
        raise ValueError("defect_visible phải là true hoặc false.")
    if types == {"normal"}:
        if visibility == "true":
            raise ValueError("Chai bình thường không thể có defect_visible=true.")
        return "normal", []
    if types == {"unknown"}:
        if visibility == "true":
            raise ValueError("Lỗi unknown cần được gán loại trước khi đánh dấu nhìn thấy.")
        return "uncertain", []
    if not types or not types <= TYPES:
        raise ValueError("defect_type không hợp lệ; dùng normal, unknown hoặc các mã lỗi đã hỗ trợ, ngăn bằng dấu ;.")
    return ("defect" if visibility == "true" else "uncertain"), sorted(types)


def validate_rows(rows, images_dir, seed=42):
    """Every bottle AND capture session stays in one split, including hidden views."""
    images_dir = Path(images_dir).resolve()
    samples, seen = [], set()
    parent = {}
    def find(value):
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]
    def union(a, b):
        parent[find(b)] = find(a)
    for index, source in enumerate(rows, 2):
        row = {key: (value or "").strip() for key, value in source.items() if key is not None}
        if any(not row.get(key) for key in REQUIRED):
            raise ValueError(f"Dòng {index}: thiếu trường bắt buộc.")
        if row["view"] not in VIEWS or row.get("split", "") not in {"", "train", "val", "test"}:
            raise ValueError(f"Dòng {index}: view hoặc split không hợp lệ.")
        label, types = visible_label(row["defect_type"], row["defect_visible"].lower())
        path = (images_dir / row["path"]).resolve()
        path.relative_to(images_dir)
        pixels = read_bgr(path)
        if min(pixels.shape[:2]) < 48 or pixels.shape[0] * pixels.shape[1] > 25_000_000:
            raise ValueError(f"Dòng {index}: ảnh cần ít nhất 48 pixel mỗi chiều và không quá 25 MP.")
        digest = hashlib.sha256(str(pixels.shape).encode("ascii") + pixels.tobytes()).hexdigest()
        if digest in seen:
            raise ValueError(f"Dòng {index}: ảnh trùng nội dung pixel; bỏ bản sao trước khi chia tập.")
        seen.add(digest)
        union("bottle:" + row["bottle_id"], "session:" + row["session_id"])
        samples.append({**row, "source_path": str(path), "id": digest[:20], "pixel_sha256": digest,
                        "label": label, "defect_types": types,
                        "defect_visible": row["defect_visible"].lower() == "true"})
    if not samples:
        raise ValueError("CSV chưa có ảnh thật. Điền dữ liệu vào các dòng dưới tiêu đề.")
    groups, explicit = {}, {}
    for sample in samples:
        component = find("bottle:" + sample["bottle_id"])
        group = "multiview:" + hashlib.sha256(component.encode("utf-8")).hexdigest()[:16]
        sample["group"] = group
        groups.setdefault(group, []).append(sample)
        if sample.get("split"):
            previous = explicit.setdefault(group, sample["split"])
            if previous != sample["split"]:
                raise ValueError("Cùng bottle_id hoặc session_id xuất hiện trong nhiều split; không được để rò rỉ dữ liệu.")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    automatic = {}
    for index, key in enumerate(keys):
        # Reserve distinct groups for validation/test when at least three exist.
        automatic[key] = "val" if len(keys) >= 3 and index == 0 else "test" if len(keys) >= 3 and index == 1 else "train"
        if len(keys) >= 10:
            automatic[key] = "val" if index < max(1, round(len(keys) * 0.15)) else "test" if index < max(2, round(len(keys) * 0.30)) else "train"
    for sample in samples:
        sample["split"] = explicit.get(sample["group"], automatic[sample["group"]])
    return samples


def import_collection(csv_path, images_dir, destination=DESTINATION, seed=42, validate_only=False):
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED <= set(reader.fieldnames or []):
            raise ValueError("CSV thiếu cột: " + ", ".join(sorted(REQUIRED - set(reader.fieldnames or []))))
        samples = validate_rows(list(reader), images_dir, seed)
    report = {"status": "awaiting_review", "source": "user_real_multiview_captures", "seed": seed,
              "image_count": len(samples), "bottle_count": len({s["bottle_id"] for s in samples}),
              "session_count": len({s["session_id"] for s in samples}),
              "counts": {split: dict(Counter(s["label"] for s in samples if s["split"] == split)) for split in ("train", "val", "test")},
              "view_counts": dict(Counter(s["view"] for s in samples)),
              "note": "Staging only; not included in training. Hidden defects are uncertain, never normal. Review images/labels, near-duplicates, viewpoint coverage and pixel-only crops before creating a training manifest. No synthetic image is claimed to be a new photograph."}
    if not validate_only:
        destination = Path(destination).resolve()
        destination.relative_to(APP_DIR.resolve())
        images = destination / "images"
        images.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            image = read_bgr(sample.pop("source_path"))
            target = images / f"{sample['id']}.png"
            if target.exists():
                if not np.array_equal(read_bgr(target), image):
                    raise ValueError("Trùng ID với ảnh đã lưu khác nội dung.")
            else:
                ok, encoded = cv2.imencode(".png", image)
                if not ok:
                    raise ValueError("Không lưu được ảnh thu thập.")
                encoded.tofile(str(target))
            sample["path"] = target.relative_to(APP_DIR).as_posix()
        report["samples"] = samples
        _atomic_json(destination / "collection.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, ensure_ascii=True, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    import_collection(args.csv, args.images_dir, args.destination, args.seed, args.validate_only)
