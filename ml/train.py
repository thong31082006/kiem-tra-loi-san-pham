"""Data validation, progress and metrics shared by the HOG + SVM workflow."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .features import APP_DIR, CLASS_NAMES, image_path

DEFAULT_MANIFEST = APP_DIR / "data" / "training" / "manifest.json"


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05)


class Progress:
    def __init__(self, artifact_root, run_id):
        self.path = Path(artifact_root) / "training_progress.json"
        self.run_id = run_id
        self.started_at = datetime.now(timezone.utc).isoformat()

    def update(self, message, state="running", model=None, **values):
        payload = {
            "state": state, "running": state == "running", "run_id": self.run_id,
            "started_at": self.started_at, "updated_at": datetime.now(timezone.utc).isoformat(),
            "model": model, "message": message, **values,
        }
        _atomic_json(self.path, payload)
        print(message, flush=True)


def load_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest chưa có records để huấn luyện.")
    seen_ids, seen_paths, group_splits = set(), set(), {}
    counts = {split: {label: 0 for label in CLASS_NAMES} for split in ("train", "val", "test")}
    for row in records:
        if row.get("label") not in CLASS_NAMES or row.get("split") not in counts:
            raise ValueError(f"Nhãn hoặc split không hợp lệ: {row.get('id')}")
        record_id = str(row.get("id", ""))
        if not record_id or record_id in seen_ids:
            raise ValueError("Mỗi ảnh trong manifest cần một id duy nhất.")
        seen_ids.add(record_id)
        path_value = image_path(row["path"]).resolve()
        path_value.relative_to(APP_DIR.resolve())
        if not path_value.is_file():
            raise FileNotFoundError(f"Không tìm thấy ảnh huấn luyện: {path_value}")
        normalized_path = os.path.normcase(str(path_value))
        if normalized_path in seen_paths:
            raise ValueError(f"Ảnh bị lặp trong manifest: {row['path']}")
        seen_paths.add(normalized_path)
        if manifest.get("input_mode") == "auto_bottle_roi" and not row.get("training_path"):
            raise ValueError(f"Ảnh chưa được chuẩn bị vùng chai tự động: {record_id}")
        if row.get("training_path"):
            training_path = image_path(row["training_path"]).resolve()
            training_path.relative_to(APP_DIR.resolve())
            if not training_path.is_file():
                raise FileNotFoundError(f"Không tìm thấy vùng chai đã chuẩn bị: {training_path}")
        group = str(row.get("group", ""))
        if not group:
            raise ValueError(f"Ảnh chưa có group chống rò rỉ dữ liệu: {record_id}")
        previous_split = group_splits.setdefault(group, row["split"])
        if previous_split != row["split"]:
            raise ValueError(f"Group xuất hiện trong nhiều split: {group}")
        counts[row["split"]][row["label"]] += 1
    for split, labels in counts.items():
        if any(count < (2 if split == "train" else 1) for count in labels.values()):
            raise ValueError(f"Tập {split} cần có đủ ảnh của cả hai lớp normal/defect: {labels}")
    return manifest, records, counts


def _metrics(labels, defect_probabilities, threshold):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    predictions = (np.asarray(defect_probabilities) >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "class_order": list(CLASS_NAMES),
        "positive_class": "defect", "sample_count": int(len(labels)),
    }


def _choose_threshold(labels, probabilities):
    # Every candidate is assessed on validation only. Prefer a central threshold on ties.
    best = None
    for threshold in np.linspace(0.1, 0.9, 33):
        scores = _metrics(labels, probabilities, float(threshold))
        rank = (scores["f1"], scores["balanced_accuracy"], -abs(float(threshold) - 0.5))
        if best is None or rank > best[0]:
            best = (rank, float(threshold), scores)
    return best[1], best[2]
