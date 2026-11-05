"""Local web application: trained model inference from one image, with no reference input."""
import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ml.inference import MODEL_NAMES, ModelService
from web_history import HistoryStore

APP_DIR = Path(__file__).resolve().parent
MANIFEST = APP_DIR / "data" / "training" / "manifest.json"
SCOPE = (
    "Mô hình thử nghiệm cho mẫu chai trong bộ dữ liệu Ron 88. "
    "Chưa được kiểm chứng cho mọi loại chai, mọi góc chụp hoặc lỗi bị che khuất."
)
MAX_UPLOAD = 15 * 1024 * 1024


def read_manifest(path=MANIFEST):
    if not path.is_file():
        return {"records": [], "counts": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_summary(manifest):
    records = manifest.get("records", [])
    counts = {split: sum(row["split"] == split for row in records) for split in ("train", "val", "test")}
    return {
        "image_count": len(records), **counts, "counts": manifest.get("counts", counts),
        "source": manifest.get("source_url", manifest.get("source", "Kaggle")),
        "scope": SCOPE, "split_note": manifest.get("split_note", ""),
    }


class TrainingOptions(BaseModel):
    epochs: int = Field(default=8, ge=1, le=30)
    batch_size: int = Field(default=8, ge=1, le=16)


class TrainingManager:
    def __init__(self, artifact_root, manifest_path):
        self.artifact_root, self.manifest_path = Path(artifact_root), Path(manifest_path)
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
            if read_manifest(self.manifest_path).get("input_mode") != "auto_bottle_roi":
                raise HTTPException(409, "Chạy scripts/prepare_bottle_crops.py để chuẩn bị vùng chai giống cách web xử lý trước khi huấn luyện.")
            self.artifact_root.mkdir(parents=True, exist_ok=True)
            self.log_path = self.artifact_root / f"web_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            command = [sys.executable, "-u", str(APP_DIR / "scripts" / "train_and_evaluate.py"),
                       "--manifest", str(self.manifest_path), "--epochs", str(options.epochs),
                       "--batch-size", str(options.batch_size), "--device", "auto",
                       "--artifact-root", str(self.artifact_root)]
            environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "4",
                           "TORCH_HOME": str(APP_DIR / "data" / "torch_cache")}
            with self.log_path.open("w", encoding="utf-8") as output:
                self.process = subprocess.Popen(
                    command, cwd=APP_DIR, env=environment, stdout=output, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            self.state, self.message = "running", "Đang huấn luyện ba mô hình. Bạn có thể theo dõi tại tab Mô hình."
            return {"state": self.state, "running": True, "message": self.message}


def create_app(service=None, history=None, manifest_path=None):
    application = FastAPI(title="Kiểm tra chai nhựa — Nhóm 8", version="2.0")
    service = service or ModelService()
    history = history or HistoryStore()
    manifest_path = Path(manifest_path or MANIFEST)
    manager = TrainingManager(service.artifact_root, manifest_path)
    prediction_lock = threading.Lock()
    application.state.model_service = service
    application.state.history = history
    application.state.training = manager
    application.mount("/static", StaticFiles(directory=APP_DIR / "web"), name="static")

    def status_data():
        status = service.status()
        dataset = dataset_summary(read_manifest(manifest_path))
        return {**status, "dataset": dataset, "training": manager.status(), "scope": SCOPE}

    @application.get("/", include_in_schema=False)
    def homepage():
        return FileResponse(APP_DIR / "web" / "index.html")

    @application.get("/api/status")
    def status():
        return status_data()

    @application.get("/api/models")
    def models():
        status = status_data()
        report = getattr(service, "_report", None) or {}
        end_to_end = {}
        if status.get("run_id"):
            path = service.artifact_root / status["run_id"] / "end_to_end.json"
            if path.is_file():
                end_to_end = json.loads(path.read_text(encoding="utf-8"))
                end_to_end.pop("records", None)
        return {**status, "report": report, "end_to_end": end_to_end}

    @application.get("/api/report.json")
    def download_report():
        data = models()
        if not data.get("ready"):
            raise HTTPException(409, "Chưa có báo cáo huấn luyện hoàn tất.")
        return JSONResponse({"training": data["report"], "end_to_end": data["end_to_end"]}, headers={
            "Content-Disposition": 'attachment; filename="bottle_training_report.json"'
        })

    @application.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @application.post("/api/train", status_code=202)
    def train(options: TrainingOptions):
        return manager.start(options)

    def predict_sync(image, model):
        if not prediction_lock.acquire(blocking=False):
            raise HTTPException(409, "Đang kiểm tra ảnh khác. Hãy thử lại sau giây lát.")
        try:
            return service.predict(image, model=model)
        finally:
            prediction_lock.release()

    @application.post("/api/predict")
    async def predict(file: UploadFile = File(...), model: str = Form("mobilenet")):
        if model not in MODEL_NAMES:
            raise HTTPException(422, "Mô hình không hợp lệ.")
        data = await file.read(MAX_UPLOAD + 1)
        await file.close()
        if not data or len(data) > MAX_UPLOAD:
            raise HTTPException(413, "Chọn một ảnh có dung lượng không quá 15 MB.")
        # Check dimensions before OpenCV allocates a decoded full-size image.
        from io import BytesIO
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(BytesIO(data)) as header:
                width, height = header.size
                if min(width, height) < 48 or width * height > 25_000_000:
                    raise HTTPException(422, "Ảnh cần tối thiểu 48 pixel mỗi chiều và không quá 25 megapixel.")
                header.verify()
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
            raise HTTPException(422, "Tệp không phải ảnh hợp lệ hoặc ảnh đã hỏng.")
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(422, "Không đọc được ảnh. Hãy chọn JPG, PNG, WebP hoặc BMP.")
        try:
            result = await run_in_threadpool(predict_sync, image, model)
        except FileNotFoundError as error:
            raise HTTPException(409, str(error))
        except (ValueError, RuntimeError) as error:
            raise HTTPException(422, str(error))
        item_id = uuid.uuid4().hex
        filename = Path((file.filename or "anh_chai.jpg").replace("\\", "/")).name[:200]
        item = {**result, "id": item_id, "filename": filename,
                "created_at": datetime.now(timezone.utc).isoformat(), "image_url": f"/api/images/{item_id}"}
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise HTTPException(500, "Không lưu được ảnh kết quả.")
        destination = history.image_path(item_id)
        try:
            encoded.tofile(str(destination))
            history.save(item)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return item

    @application.get("/api/history")
    def list_history(limit: int = Query(100, ge=1, le=500)):
        return {"items": history.list(limit)}

    @application.delete("/api/history/{item_id}")
    def delete_history(item_id: str):
        if not history.get(item_id):
            raise HTTPException(404, "Không tìm thấy lần kiểm tra.")
        history.delete(item_id)
        return {"deleted": True}

    @application.get("/api/images/{item_id}")
    def get_image(item_id: str):
        if not history.get(item_id):
            raise HTTPException(404, "Không tìm thấy ảnh.")
        path = history.image_path(item_id)
        if not path.is_file():
            raise HTTPException(404, "Tệp ảnh không còn tồn tại.")
        return FileResponse(path, media_type="image/jpeg")

    @application.get("/api/samples")
    def samples(split: str = "test", label: str = "all", limit: int = Query(60, ge=1, le=1500), offset: int = Query(0, ge=0)):
        if split not in ("train", "val", "test") or label not in ("all", "normal", "defect"):
            raise HTTPException(422, "Bộ lọc dữ liệu không hợp lệ.")
        records = [row for row in read_manifest(manifest_path).get("records", [])
                   if row["split"] == split and (label == "all" or row["label"] == label)]
        return {"total": len(records), "samples": [
            {"id": row["id"], "image_url": f"/api/samples/{row['id']}/image", "label": row["label"],
             "split": row["split"], "filename": Path(row["path"]).name, "defect_types": row.get("defect_types", [])}
            for row in records[offset:offset + limit]
        ]}

    @application.get("/api/samples/{sample_id}/image")
    def sample_image(sample_id: str):
        row = next((row for row in read_manifest(manifest_path).get("records", []) if row["id"] == sample_id), None)
        if row is None:
            raise HTTPException(404, "Không tìm thấy ảnh mẫu.")
        path = (APP_DIR / row["path"]).resolve()
        try:
            path.relative_to((APP_DIR / "data" / "datasets").resolve())
        except ValueError:
            raise HTTPException(404, "Đường dẫn ảnh mẫu không hợp lệ.")
        if not path.is_file():
            raise HTTPException(404, "Tệp ảnh mẫu không còn tồn tại.")
        return FileResponse(path, media_type="image/jpeg")

    return application


app = create_app()
