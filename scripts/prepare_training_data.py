"""Audit explicit defect annotations and create one shared, reproducible three-way split."""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = APP_DIR / "data" / "datasets" / "bottle_defects"
MANIFEST = APP_DIR / "data" / "training" / "manifest.json"
SOURCE_URL = "https://www.kaggle.com/datasets/jihadakbr/water-bottle-defect-level-detection-dataset"
CLASSES = {0: "bottle_ron88", 1: "bottle_other_brand", 2: "defect_low_fill", 3: "defect_no_cap",
           4: "defect_loose_cap", 5: "defect_debris", 6: "defect_label_damage"}
CAPTURE = re.compile(r"^(?P<category>.+)_(?P<index>\d+)_(?P<day>\d{8})_(?P<time>\d{6})$")


def parse_labels(path):
    if not path.is_file():
        raise ValueError(f"Missing annotation: {path}")
    result = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO detection annotation: {path}")
        category = int(values[0])
        coordinates = [float(value) for value in values[1:]]
        if category not in CLASSES or not all(np.isfinite(value) and 0 <= value <= 1 for value in coordinates) or min(coordinates[2:]) <= 0:
            raise ValueError(f"Invalid class or coordinates: {path}")
        result.append(category)
    return result


def audit(source_dir=SOURCE_DIR, seed=42, block_size=40, guard=3):
    source_dir = Path(source_dir).resolve()
    candidates, excluded, seen_hashes = [], [], {}
    annotation_counts = Counter()
    dimensions = Counter()
    images = sorted((source_dir / "images").rglob("*.jpg"))
    if not images:
        raise ValueError("Chưa có bộ ảnh lỗi. Chạy scripts/download_defect_dataset.py trước.")
    for position, path in enumerate(images):
        relative = path.relative_to(source_dir / "images")
        label_path = (source_dir / "labels" / relative).with_suffix(".txt")
        categories = parse_labels(label_path)
        annotation_counts.update(CLASSES[value] for value in categories)
        reason = None
        match = CAPTURE.match(path.stem)
        if not match:
            raise ValueError(f"Không nhận diện được nhóm ảnh chụp: {path.name}")
        capture = match.groupdict()
        index = int(capture["index"])
        category = capture["category"]
        defects = sorted({CLASSES[value] for value in categories if value >= 2})
        if 1 in categories:
            reason = "other_brand_excluded_from_binary_quality_task"
        elif 0 not in categories:
            reason = "missing_bottle_annotation"
        elif not defects and category != "good":
            reason = "non_good_capture_without_explicit_defect_annotation"
        # Separate adjacent blocks by six omitted frames. No augmentation is done here.
        elif index % block_size < guard or index % block_size >= block_size - guard:
            reason = "temporal_boundary_guard"
        if reason:
            excluded.append({"path": str(path.relative_to(APP_DIR)), "reason": reason})
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        label = "defect" if defects else "normal"
        if digest in seen_hashes:
            previous = seen_hashes[digest]
            if previous["label"] != label:
                raise ValueError(f"Identical image has conflicting labels: {path.name}")
            excluded.append({"path": str(path.relative_to(APP_DIR)), "reason": "exact_duplicate"})
            continue
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {path.name}")
        h, w = image.shape[:2]
        dimensions[f"{w}x{h}"] += 1
        group = f"{category}:{capture['day']}:block{index // block_size:03d}"
        record = {"id": hashlib.sha256(relative.as_posix().encode()).hexdigest()[:20],
                  "path": path.relative_to(APP_DIR).as_posix(), "label": label, "group": group,
                  "split": None, "defect_types": defects, "source": SOURCE_URL,
                  "source_split": relative.parts[0], "capture_category": category,
                  "capture_index": index, "sha256": digest, "width": w, "height": h}
        bottle_boxes = [list(map(float, line.split()[1:])) for line in label_path.read_text(encoding="utf-8-sig").splitlines()
                        if line.strip() and int(line.split()[0]) == 0]
        if len(bottle_boxes) != 1:
            raise ValueError(f"Expected one annotated bottle: {path.name}")
        record["bbox"] = bottle_boxes[0]
        seen_hashes[digest] = record
        candidates.append(record)
        if (position + 1) % 250 == 0:
            print(f"Audited {position + 1}/{len(images)} images", flush=True)

    # Stratify by capture category, keeping every temporal block in exactly one split.
    strata = defaultdict(set)
    for record in candidates:
        strata[record["capture_category"]].add(record["group"])
    assignment = {}
    rng = random.Random(seed)
    for category, group_names in sorted(strata.items()):
        groups = sorted(group_names)
        if len(groups) < 3:
            raise ValueError(f"Need at least three capture blocks for {category}; found {len(groups)}")
        rng.shuffle(groups)
        test_count = max(1, round(len(groups) * 0.15))
        validation_count = max(1, round(len(groups) * 0.15))
        for position, group in enumerate(groups):
            assignment[group] = "test" if position < test_count else "val" if position < test_count + validation_count else "train"
    for record in candidates:
        record["split"] = assignment[record["group"]]
    counts = {split: {**dict(Counter(row["label"] for row in candidates if row["split"] == split)),
                      "total": sum(row["split"] == split for row in candidates)} for split in ("train", "val", "test")}
    for split, count in counts.items():
        if min(count.get("normal", 0), count.get("defect", 0)) < 10:
            raise ValueError(f"Not enough samples of both classes in {split}: {count}")
    groups_by_split = {split: {row["group"] for row in candidates if row["split"] == split} for split in counts}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        assert groups_by_split[first].isdisjoint(groups_by_split[second])
    return {
        "version": 2, "created_at": datetime.now(timezone.utc).isoformat(), "seed": seed, "input_mode": "bottle_roi",
        "source_url": SOURCE_URL, "source": "Water Bottle Defect-Level Detection Dataset, v1",
        "raw_image_count": len(images), "image_count": len(candidates), "class_names": ["normal", "defect"],
        "annotation_classes": CLASSES, "annotation_counts": dict(annotation_counts),
        "dimensions": dict(dimensions), "counts": counts,
        "label_rule": "normal: author good capture plus bottle_ron88 and no explicit defects; defect: at least one explicit defect class 2..6; other brands excluded",
        "split_method": {"name": "category_stratified_temporal_blocks", "block_size": block_size,
                         "guard_frames_each_side": guard, "target_ratios": [0.70, 0.15, 0.15]},
        "split_note": "Khối ảnh theo thứ tự chụp được giữ cùng tập, bỏ ảnh ở ranh giới và loại bản sao byte. Nguồn không có ID chai vật lý; đây chưa phải đánh giá trên các chai hoặc phiên chụp độc lập, và chưa loại hết ảnh gần giống nhau.",
        "scope": "Một mẫu chai Ron 88, camera và ánh sáng của bộ nguồn; không chứng minh mọi góc chụp.",
        "excluded_counts": dict(Counter(row["reason"] for row in excluded)),
        "excluded": excluded, "records": sorted(candidates, key=lambda row: row["id"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=40)
    parser.add_argument("--guard", type=int, default=3)
    args = parser.parse_args()
    if args.block_size < 10 or not 0 <= args.guard < args.block_size // 2:
        parser.error("Invalid temporal block/guard size")
    manifest = audit(seed=args.seed, block_size=args.block_size, guard=args.guard)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    print(json.dumps({"raw": manifest["raw_image_count"], "used": manifest["image_count"],
                      "counts": manifest["counts"], "excluded": manifest["excluded_counts"]}, ensure_ascii=True, indent=2))
    print(f"Ready: {MANIFEST}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Data preparation failed: {error}", file=sys.stderr)
        sys.exit(1)
