"""Validation and subprocess lifecycle for the website's HOG + SVM training job."""
import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ml.features import APP_DIR
from web_dataset import DatasetCatalog


class TrainingOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class TrainingManager:
    def __init__(self, artifact_root, manifest_path, catalog=None):
        self.artifact_root, self.manifest_path = Path(artifact_root), Path(manifest_path)
        self.catalog = catalog if catalog is not None else DatasetCatalog(self.manifest_path)
        self.lock = threading.Lock()
        self.process = None
        self.message = "Chưa có tác vụ huấn luyện từ web."
        self.state = "idle"
        self.log_path = None

    def status(self):
        progress_path = self.artifact_root / "training_progress.json"
        progress = {}
        if progress_path.is_file():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        if self.process:
            code = self.process.poll()
            if code is None:
                if progress.get("state") != "running":
                    progress = {}
                return {**progress, "state": "running", "running": True, "message": progress.get("message", self.message)}
            self.state = "complete" if code == 0 else "failed"
            self.message = "Huấn luyện hoàn tất." if code == 0 else "Huấn luyện gặp lỗi. Xem tệp nhật ký để biết chi tiết."
            if progress.get("state") != self.state:
                progress = {}
        reported_state = progress.get("state")
        state = self.state if self.process else (reported_state or ("running" if progress.get("running") else "idle"))
        return {**progress, "state": state, "running": state == "running", "message": progress.get("message", self.message)}

    def start(self, options):
        with self.lock:
            if self.status().get("running"):
                raise HTTPException(409, "Đang có một lần huấn luyện. Hãy chờ lần hiện tại kết thúc.")
            if not self.manifest_path.is_file():
                raise HTTPException(409, "Chưa chuẩn bị dữ liệu huấn luyện. Chạy scripts/prepare_training_data.py trước.")
            if self.catalog.input_mode != "auto_bottle_roi":
                raise HTTPException(409, "Chạy scripts/prepare_bottle_crops.py để chuẩn bị vùng chai giống cách web xử lý trước khi huấn luyện.")
            self.artifact_root.mkdir(parents=True, exist_ok=True)
            self.log_path = self.artifact_root / f"web_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            command = [sys.executable, "-u", str(APP_DIR / "scripts" / "train_hog_svm.py"),
                       "--manifest", str(self.manifest_path),
                       "--seed", str(options.seed),
                       "--artifact-root", str(self.artifact_root)]
            environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "4",
                           "TORCH_HOME": str(APP_DIR / "data" / "torch_cache")}
            with self.log_path.open("w", encoding="utf-8") as output:
                self.process = subprocess.Popen(
                    command, cwd=APP_DIR, env=environment, stdout=output, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            self.state, self.message = "running", "Đang huấn luyện và test HOG + SVM."
            return {"state": self.state, "running": True, "message": self.message}
