"""Independent, pixel-only six-class analysis; reference annotations never enter inference."""
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from .features import APP_DIR, as_rgb_pil, cnn_transform, extract_features, validate_image

TYPE_CODES = ("normal", "defect_low_fill", "defect_no_cap", "defect_loose_cap", "defect_debris", "defect_label_damage")
TYPE_NAMES = {
    "normal": "Không thấy lỗi trong các nhóm đã học",
    "defect_low_fill": "Thiếu mức nước",
    "defect_no_cap": "Thiếu nắp",
    "defect_loose_cap": "Nắp lỏng / hỏng",
    "defect_debris": "Dị vật nhìn thấy",
    "defect_label_damage": "Nhãn hỏng / lệch / thiếu",
}
UNSUPPORTED_TYPES = ["Móp / biến dạng chai", "Nước đục / đổi màu"]
TYPE_ROOT = APP_DIR / "models" / "defect_types"
TYPE_SCOPE = (
    "Nhận diện loại lỗi chính trên mẫu chai Ron 88. Nhãn hỏng, lệch và thiếu được nguồn gộp chung. "
    "Chưa kiểm chứng nhiều lỗi đồng thời, hãng chai khác hoặc lỗi bị khuất. "
    "Chưa hỗ trợ móp chai và nước đục; ảnh không xác nhận được độ kín thực tế của nắp."
)


def classify_type_scores(scores, policies):
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(TYPE_CODES),) or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1, atol=1e-4):
        raise ValueError("Điểm phân loại lỗi không hợp lệ.")
    index = int(values.argmax())
    code, score = TYPE_CODES[index], float(values[index])
    policy = policies.get(code, {})
    accepted = score >= float(policy.get("threshold", 1.01)) and policy.get("enabled", False)
    return {
        "status": ("normal" if index == 0 else "typed") if accepted else "uncertain",
        "code": code if accepted else None, "name": TYPE_NAMES[code] if accepted else "Chưa xác định được loại lỗi",
        "score": score, "threshold": policy.get("threshold"),
        "scores": {key: float(value) for key, value in zip(TYPE_CODES, values)},
        "scope": TYPE_SCOPE, "unsupported": UNSUPPORTED_TYPES,
    }


class DefectTypeService:
    def __init__(self, artifact_root=None):
        self.root = Path(artifact_root or TYPE_ROOT).resolve()
        self.lock = threading.RLock()
        self.run_id = None
        self.report = None
        self.loaded = None

    def _inside(self, path):
        path = (self.root / path).resolve()
        path.relative_to(self.root)
        return path

    def _refresh(self):
        pointer = self.root / "latest.json"
        if not pointer.is_file():
            return None
        run_id = json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
        if not run_id or self._inside(run_id) == self.root:
            raise ValueError("Đường dẫn bộ phân tích loại lỗi không hợp lệ.")
        if run_id != self.run_id:
            report = json.loads((self._inside(run_id) / "report.json").read_text(encoding="utf-8"))
            if report.get("status") != "complete" or report.get("class_names") != list(TYPE_CODES):
                raise ValueError("Bộ phân tích loại lỗi chưa hoàn tất hoặc sai thứ tự lớp.")
            if not self._inside(Path(run_id) / report["artifact"]).is_file():
                raise FileNotFoundError("Thiếu trọng số bộ phân tích loại lỗi.")
            self.report, self.run_id, self.loaded = report, run_id, None
        return self._inside(run_id)

    def status(self):
        with self.lock:
            try:
                run_dir = self._refresh()
                progress_path = self.root / "training_progress.json"
                progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
                return {
                    "ready": bool(run_dir), "run_id": self.run_id,
                    "model_name": (self.report or {}).get("model_name"),
                    "scope": TYPE_SCOPE, "unsupported": UNSUPPORTED_TYPES,
                    "classes": [{"code": code, "name": TYPE_NAMES[code],
                                 **(self.report or {}).get("policies", {}).get(code, {})} for code in TYPE_CODES[1:]],
                    "test": (self.report or {}).get("test", {}),
                    "counts": (self.report or {}).get("counts", {}),
                    "training": progress,
                }
            except (OSError, ValueError, KeyError) as error:
                return {"ready": False, "error": str(error), "scope": TYPE_SCOPE, "unsupported": UNSUPPORTED_TYPES}

    def predict(self, image):
        validate_image(image)
        started = time.perf_counter()
        with self.lock:
            run_dir = self._refresh()
            if not run_dir:
                return {"status": "unavailable", "name": "Chưa có bộ phân tích loại lỗi", "scope": TYPE_SCOPE}
            if self.loaded is None:
                path = self._inside(Path(self.run_id) / self.report["artifact"])
                if self.report["model_id"] == "mobilenet":
                    import torch
                    from torchvision.models import mobilenet_v3_small
                    torch.set_num_threads(min(4, os.cpu_count() or 1))
                    network = mobilenet_v3_small(weights=None)
                    network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, len(TYPE_CODES))
                    network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
                    self.loaded = (network.eval(), cnn_transform())
                else:
                    import joblib
                    self.loaded = joblib.load(path)
                    if list(self.loaded.classes_) != list(range(len(TYPE_CODES))):
                        raise ValueError("Mô hình phân tích loại lỗi thiếu lớp hoặc sai thứ tự lớp.")
            if self.report["model_id"] == "mobilenet":
                import torch
                network, transform = self.loaded
                with torch.inference_mode():
                    scores = network(transform(as_rgb_pil(image)).unsqueeze(0)).softmax(dim=1)[0].tolist()
            else:
                scores = self.loaded.predict_proba(extract_features(image).reshape(1, -1))[0]
            result = classify_type_scores(scores, self.report["policies"])
            return {**result, "model_name": self.report["model_name"], "run_id": self.run_id,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
