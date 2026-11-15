"""Load trusted local training artifacts and classify one image without a reference."""
import json
import os
import threading
import time
from pathlib import Path

from .features import APP_DIR, CLASS_NAMES, as_rgb_pil, cnn_transform, extract_features, validate_image

DEFAULT_ARTIFACT_ROOT = APP_DIR / "models" / "trained"
MODEL_NAMES = {
    "svm": "HOG + SVM",
    "random_forest": "HOG + Random Forest",
    "mobilenet": "MobileNetV3 Small",
}


def _read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _inside(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


class ModelService:
    def __init__(self, artifact_root=None, enable_defect_types=True):
        self.artifact_root = Path(artifact_root or DEFAULT_ARTIFACT_ROOT).resolve()
        self._lock = threading.RLock()
        self._run_id = None
        self._report = None
        self._loaded = {}
        self._locator = None
        self._type_service = None
        if enable_defect_types:
            from .defect_types import DefectTypeService
            self._type_service = DefectTypeService()

    def _refresh(self):
        latest = self.artifact_root / "latest.json"
        if not latest.is_file():
            return None
        pointer = _read_json(latest)
        run_id = pointer.get("run_id", "")
        run_dir = _inside(self.artifact_root, run_id)
        if not run_id or run_dir == self.artifact_root:
            raise ValueError("Đường dẫn mô hình không hợp lệ.")
        if run_id != self._run_id:
            report = _read_json(run_dir / "report.json")
            if report.get("status") != "complete":
                raise ValueError("Lần huấn luyện chưa hoàn tất.")
            self._run_id, self._report, self._loaded = run_id, report, {}
        return run_dir

    def status(self):
        with self._lock:
            try:
                run_dir = self._refresh()
                models = []
                for model_id, name in MODEL_NAMES.items():
                    entry = (self._report or {}).get("models", {}).get(model_id, {})
                    ready = bool(run_dir and entry and _inside(run_dir, entry["artifact"]).is_file())
                    models.append({
                        "id": model_id,
                        "name": name,
                        "ready": ready,
                        "metrics": entry.get("test", {}),
                        "validation": entry.get("validation", {}),
                        "threshold": entry.get("decision_threshold"),
                        "uncertainty_threshold": entry.get("uncertainty_threshold", 0.65),
                        "training_seconds": entry.get("training_seconds"),
                        "inference_ms": entry.get("inference_ms"),
                        "curves": entry.get("curves", []),
                    })
                progress = self.artifact_root / "training_progress.json"
                return {
                    "ready": bool(models and all(model["ready"] for model in models)),
                    "run_id": self._run_id,
                    "models": models,
                    "class_names": list(CLASS_NAMES),
                    "dataset": (self._report or {}).get("dataset", {}),
                    "training": _read_json(progress) if progress.is_file() else {"running": False},
                    "evaluation_note": (self._report or {}).get("evaluation_note", ""),
                    "defect_types": self._type_service.status() if self._type_service else {"ready": False},
                }
            except (OSError, ValueError, KeyError) as error:
                return {"ready": False, "run_id": self._run_id, "models": [], "error": str(error)}

    def _load(self, model_id, run_dir, entry):
        if model_id not in self._loaded:
            artifact = _inside(run_dir, entry["artifact"])
            if model_id == "mobilenet":
                import torch
                from torchvision.models import mobilenet_v3_small

                torch.set_num_threads(min(4, os.cpu_count() or 1))
                network = mobilenet_v3_small(weights=None)
                network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, len(CLASS_NAMES))
                checkpoint = torch.load(artifact, map_location="cpu", weights_only=True)
                network.load_state_dict(checkpoint["state_dict"])
                network.eval()
                self._loaded[model_id] = (network, cnn_transform())
            else:
                import joblib

                self._loaded[model_id] = joblib.load(artifact)
        return self._loaded[model_id]

    def predict(self, image_bgr, model="mobilenet"):
        validate_image(image_bgr)
        if model not in MODEL_NAMES:
            raise ValueError("Thuật toán không được hỗ trợ.")
        started = time.perf_counter()
        with self._lock:
            run_dir = self._refresh()
            if not run_dir:
                raise FileNotFoundError("Chưa có mô hình đã huấn luyện. Chạy scripts/train_models.py trước.")
            entry = self._report["models"].get(model)
            if not entry:
                raise FileNotFoundError("Chưa có mô hình cho thuật toán đã chọn.")
            cold_start = model not in self._loaded
            loaded = self._load(model, run_dir, entry)
            localization = None
            if self._report.get("input_mode") in ("bottle_roi", "auto_bottle_roi"):
                if self._locator is None:
                    from .localization import BottleLocator
                    self._locator = BottleLocator()
                image_bgr, localization = self._locator.locate(image_bgr)
            if model == "mobilenet":
                import torch

                network, transform = loaded
                tensor = transform(as_rgb_pil(image_bgr)).unsqueeze(0)
                with torch.inference_mode():
                    probabilities = network(tensor).softmax(dim=1)[0].tolist()
                defect_probability = float(probabilities[1])
            else:
                vector = extract_features(image_bgr).reshape(1, -1)
                probabilities = loaded.predict_proba(vector)[0]
                classes = list(loaded.classes_)
                defect_probability = float(probabilities[classes.index(1)])
            decision_threshold = float(entry.get("decision_threshold", 0.5))
            uncertainty_threshold = float(entry.get("uncertainty_threshold", 0.65))
            raw_label = "defect" if defect_probability >= decision_threshold else "normal"
            confidence = defect_probability if raw_label == "defect" else 1.0 - defect_probability
            label = raw_label if confidence >= uncertainty_threshold else "uncertain"
            type_analysis = None
            if self._type_service:
                try:
                    type_analysis = self._type_service.predict(image_bgr)
                    type_analysis["agreement"] = (
                        "agree" if (label == "normal" and type_analysis["status"] == "normal")
                        or (label == "defect" and type_analysis["status"] == "typed") else "review"
                    )
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    type_analysis = {"status": "unavailable", "name": "Bộ phân tích loại lỗi tạm thời chưa sẵn sàng", "error": str(error)}
            return {
                "label": label,
                "raw_label": raw_label,
                "confidence": confidence,
                "probabilities": {"normal": 1.0 - defect_probability, "defect": defect_probability},
                "model": model,
                "model_name": MODEL_NAMES[model],
                "run_id": self._run_id,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "cold_start": cold_start,
                "threshold": decision_threshold,
                "uncertainty_threshold": uncertainty_threshold,
                "confidence_note": "Xác suất mô hình, chưa hiệu chuẩn thành độ tin cậy thực tế.",
                "localization": localization,
                "defect_analysis": type_analysis,
            }
