"""One six-class HOG + SVM supplies both quality and defect-type predictions.

Train and inference share extract_hog. No filename or annotation enters predict.
The existing SSDLite locator only crops the bottle; it does not classify defects.
"""
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .defect_types import TYPE_CODES, TYPE_NAMES, classify_type_scores
from .features import APP_DIR, validate_image

ARTIFACT_ROOT = APP_DIR / "models" / "hog_svm"
FEATURE_VERSION = "auto-bottle-roi-hog64x128-3780-v1"
MODEL_NAMES = {"svm": "HOG + SVM"}


def extract_hog(image):
    validate_image(image)
    gray = cv2.cvtColor(cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    descriptor = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    return descriptor.compute(gray).reshape(-1).astype(np.float32)


def quality_result(scores, threshold, uncertainty=0.65):
    probability = float(np.sum(scores[1:]))
    raw = "defect" if probability >= threshold else "normal"
    confidence = probability if raw == "defect" else float(scores[0])
    return {"label": raw if confidence >= uncertainty else "uncertain", "raw_label": raw,
            "confidence": confidence, "probabilities": {"normal": float(scores[0]), "defect": probability},
            "threshold": threshold, "uncertainty_threshold": uncertainty}


class HogSvmService:
    supported_models = MODEL_NAMES

    def __init__(self, artifact_root=None, run_id=None):
        self.artifact_root = Path(artifact_root or ARTIFACT_ROOT).resolve()
        self.pinned_run_id = run_id
        self._lock = threading.RLock()
        self._run_id = self._report = self._loaded = self._locator = None

    def _refresh(self):
        pointer = self.artifact_root / "latest.json"
        if not self.pinned_run_id and not pointer.is_file():
            return None
        run_id = self.pinned_run_id or json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
        run_dir = (self.artifact_root / run_id).resolve()
        run_dir.relative_to(self.artifact_root)
        if run_dir == self.artifact_root:
            raise ValueError("Đường dẫn mô hình không hợp lệ.")
        if run_id != self._run_id:
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            if (report.get("status") != "complete" or report.get("feature_version") != FEATURE_VERSION
                    or report.get("class_names") != list(TYPE_CODES)):
                raise ValueError("Mô hình chưa hoàn tất hoặc sai phiên bản đặc trưng/lớp.")
            if not (run_dir / "svm.joblib").is_file():
                raise FileNotFoundError("Thiếu trọng số HOG + SVM.")
            self._run_id, self._report, self._loaded = run_id, report, None
        return run_dir

    def status(self):
        with self._lock:
            try:
                ready = bool(self._refresh())
                report = self._report or {}
                entry = report.get("models", {}).get("svm", {})
                types = report.get("defect_types", {})
                return {"ready": ready, "run_id": self._run_id,
                        "models": [{"id": "svm", "name": "HOG + SVM", "ready": ready,
                                    "metrics": entry.get("test", {}), "validation": entry.get("validation", {})}],
                        "dataset": report.get("dataset", {}), "evaluation_note": report.get("evaluation_note", ""),
                        "defect_types": {"ready": ready, "model_name": "Cùng mô hình HOG + SVM 6 lớp",
                                         "classes": [{"code": code, "name": TYPE_NAMES[code], **report.get("policies", {}).get(code, {})}
                                                     for code in TYPE_CODES[1:]], **types}}
            except (OSError, ValueError, KeyError) as error:
                return {"ready": False, "models": [], "error": str(error)}

    def predict(self, image_bgr, model="svm"):
        if model != "svm":
            raise ValueError("Phiên bản này sử dụng HOG + SVM.")
        validate_image(image_bgr)
        started = time.perf_counter()
        with self._lock:
            run_dir = self._refresh()
            if not run_dir:
                raise FileNotFoundError("Chạy scripts/train_hog_svm.py để huấn luyện HOG + SVM trước.")
            cold_start = self._loaded is None
            if cold_start:
                import joblib
                self._loaded = joblib.load(run_dir / "svm.joblib")
                if list(self._loaded.classes_) != list(range(len(TYPE_CODES))):
                    raise ValueError("Mô hình không đủ 6 lớp theo thứ tự đã huấn luyện.")
            if self._locator is None:
                from .localization import BottleLocator
                self._locator = BottleLocator()
            crop, localization = self._locator.locate(image_bgr)
            scores = self._loaded.predict_proba(extract_hog(crop).reshape(1, -1))[0]
            entry = self._report["models"]["svm"]
            quality = quality_result(scores, entry["decision_threshold"], entry["uncertainty_threshold"])
            kind = classify_type_scores(scores, self._report["policies"])
            kind.update({"model_name": "HOG + SVM (cùng mô hình 6 lớp)", "run_id": self._run_id,
                         "agreement": "agree" if (quality["label"] == "normal" and kind["status"] == "normal")
                         or (quality["label"] == "defect" and kind["status"] == "typed") else "review"})
            return {**quality, "model": "svm", "model_name": "HOG + SVM", "run_id": self._run_id,
                    "cold_start": cold_start, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "localization": localization, "orientation": None, "defect_analysis": kind,
                    "confidence_note": "Điểm mô hình chưa hiệu chuẩn thành xác suất đúng thực tế."}
