"""Local web application: trained model inference from one image, with no reference input."""
import json
import os
import subprocess
import sys
import threading
import uuid
import csv
import io
import hashlib
from typing import Literal
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool

from ml.inference import MODEL_NAMES, ModelService
from web_history import HistoryStore
from web_visuals import build_visuals

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


class ReviewOptions(BaseModel):
    target: Literal["quality", "defect_type"] = "quality"
    verdict: Literal["correct", "incorrect", "unsure"]
    corrected_label: Literal["normal", "defect", "uncertain"] | None = None
    corrected_type: Literal["normal", "defect_low_fill", "defect_no_cap", "defect_loose_cap", "defect_debris", "defect_label_damage", "defect_dent", "defect_cloudy", "other", "unknown"] | None = None
    note: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def validate_correction(self):
        if self.target == "quality" and self.corrected_type is not None:
            raise ValueError("Đánh giá chất lượng chỉ nhận nhãn chất lượng được sửa.")
        if self.target == "defect_type" and self.corrected_label is not None:
            raise ValueError("Đánh giá loại lỗi chỉ nhận loại lỗi được sửa.")
        if self.verdict != "incorrect" and (self.corrected_label is not None or self.corrected_type is not None):
            raise ValueError("Chỉ ghi nhãn sửa khi đánh dấu kết quả sai.")
        return self


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
        return JSONResponse({"training": data["report"], "end_to_end": data["end_to_end"], "defect_types": data.get("defect_types", {})}, headers={
            "Content-Disposition": 'attachment; filename="bottle_training_report.json"'
        })

    @application.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @application.post("/api/train", status_code=202)
    def train(options: TrainingOptions):
        if not prediction_lock.acquire(blocking=False):
            raise HTTPException(409, "Đang kiểm tra ảnh. Hãy chờ xử lý xong trước khi huấn luyện.")
        try:
            return manager.start(options)
        finally:
            prediction_lock.release()

    def predict_sync(image, model_ids, tolerate_errors=False):
        if not prediction_lock.acquire(blocking=False):
            raise HTTPException(409, "Đang kiểm tra ảnh khác. Hãy thử lại sau giây lát.")
        try:
            results = []
            for model_id in model_ids:
                try:
                    results.append(service.predict(image, model=model_id))
                except (FileNotFoundError, ValueError, RuntimeError) as error:
                    if not tolerate_errors:
                        raise
                    results.append({"model": model_id, "model_name": MODEL_NAMES[model_id], "error": str(error)})
            return results
        finally:
            prediction_lock.release()

    async def decode_upload(file):
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
        return image

    def save_prediction(image, result, filename, batch=None):
        item_id = uuid.uuid4().hex
        filename = Path((filename or "anh_chai.jpg").replace("\\", "/")).name[:200]
        item = {**result, "id": item_id, "filename": filename,
                "created_at": datetime.now(timezone.utc).isoformat(), "image_url": f"/api/images/{item_id}"}
        if batch:
            item.update({"batch_id": batch["batch_id"], "batch_position": batch["position"]})
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise HTTPException(500, "Không lưu được ảnh kết quả.")
        destination = history.image_path(item_id)
        try:
            encoded.tofile(str(destination))
            history.save(item, batch=batch)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return item

    @application.post("/api/predict")
    async def predict(file: UploadFile = File(...), model: str = Form("mobilenet")):
        if model not in MODEL_NAMES:
            raise HTTPException(422, "Mô hình không hợp lệ.")
        image = await decode_upload(file)
        try:
            result = (await run_in_threadpool(predict_sync, image, [model]))[0]
        except FileNotFoundError as error:
            raise HTTPException(409, str(error))
        except (ValueError, RuntimeError) as error:
            raise HTTPException(422, str(error))
        analysis = await run_in_threadpool(build_visuals, image, result.get("localization"))
        item = await run_in_threadpool(save_prediction, image, result, file.filename)
        return {**item, "analysis": analysis}

    @application.post("/api/compare")
    async def compare(file: UploadFile = File(...)):
        image = await decode_upload(file)
        results = await run_in_threadpool(predict_sync, image, list(MODEL_NAMES), True)
        valid = [row for row in results if "error" not in row]
        if not valid:
            raise HTTPException(422, results[0]["error"])
        analysis = await run_in_threadpool(build_visuals, image, valid[0].get("localization"))
        items = []
        for row in results:
            items.append(row if "error" in row else await run_in_threadpool(save_prediction, image, row, file.filename))
        return {"results": items, "analysis": analysis}

    @application.post("/api/batch/predict")
    async def batch_predict(file: UploadFile = File(...), model: str = Form("svm"),
                            batch_id: str = Form(..., pattern=r"^[a-f0-9]{32}$"),
                            position: int = Form(..., ge=0, le=49)):
        if model not in MODEL_NAMES:
            raise HTTPException(422, "Mô hình không hợp lệ.")
        image = await decode_upload(file)
        content_hash = hashlib.sha256(str(image.shape).encode("ascii") + image.tobytes()).hexdigest()
        batch = {"batch_id": batch_id, "position": position, "content_hash": content_hash}

        def process():
            if not prediction_lock.acquire(blocking=False):
                raise HTTPException(409, "Đang xử lý ảnh khác. Hãy thử lại sau giây lát.")
            try:
                previous = history.batch_result(batch_id, position)
                if previous and previous["item"]:
                    if previous["content_hash"] != content_hash or previous["model"] != model:
                        raise HTTPException(409, "Vị trí trong lô đã chứa ảnh hoặc mô hình khác. Hãy tạo lô mới.")
                    item = previous["item"]
                else:
                    if manager.status().get("running"):
                        raise HTTPException(409, "Đang huấn luyện mô hình. Hãy tạm dừng lô ảnh và thử lại khi hoàn tất.")
                    result = service.predict(image, model=model)
                    item = save_prediction(image, result, file.filename, batch)
                return {**item, "analysis": build_visuals(image, item.get("localization"))}
            finally:
                prediction_lock.release()

        try:
            return await run_in_threadpool(process)
        except FileNotFoundError as error:
            raise HTTPException(409, str(error))
        except (ValueError, RuntimeError) as error:
            raise HTTPException(422, str(error))

    @application.get("/api/history.csv")
    def export_history():
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["Thời điểm (UTC)", "Tên ảnh", "Mô hình", "Kết quả", "Điểm mô hình", "Thời gian (ms)", "AI phân tích loại lỗi", "Đối chiếu hai bộ phân tích", "Mã lô", "Đánh giá chất lượng", "Nhãn chất lượng sửa", "Ghi chú chất lượng", "Đánh giá loại lỗi", "Loại lỗi sửa", "Ghi chú loại lỗi"])
        labels = {"normal": "Bình thường", "defect": "Có dấu hiệu lỗi", "uncertain": "Chưa đủ tin cậy"}
        for item in history.list(500):
            filename = item.get("filename", "")
            # A user-supplied filename must remain text when opened in Excel.
            if filename.lstrip().startswith(("=", "+", "-", "@")):
                filename = "'" + filename
            analysis = item.get("defect_analysis") or {}
            reviews = item.get("reviews", {})
            quality, kind = reviews.get("quality", {}), reviews.get("defect_type", {})
            cells = [item["created_at"], filename, item.get("model_name", item.get("model")),
                             labels.get(item["label"], item["label"]), item.get("confidence"), item.get("latency_ms"),
                             analysis.get("name", "Chưa có"), analysis.get("agreement", ""), item.get("batch_id", ""),
                             quality.get("verdict", ""), quality.get("corrected_label", ""), quality.get("note", ""),
                             kind.get("verdict", ""), kind.get("corrected_type", ""), kind.get("note", "")]
            writer.writerow(["'" + cell if isinstance(cell, str) and cell.lstrip().startswith(("=", "+", "-", "@")) else cell for cell in cells])
        return Response("\ufeff" + output.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="lich_su_kiem_tra.csv"'})

    @application.get("/api/history")
    def list_history(limit: int = Query(100, ge=1, le=500), review: Literal["all", "unreviewed", "correct", "incorrect", "unsure"] = "all"):
        return {"items": history.list(limit, review=review)}

    @application.get("/api/history/{item_id}")
    def get_history_item(item_id: str):
        item = history.get(item_id)
        if item is None:
            raise HTTPException(404, "Không tìm thấy lần kiểm tra.")
        return item

    @application.put("/api/history/{item_id}/review")
    def review_prediction(item_id: str, options: ReviewOptions):
        item = history.get(item_id)
        if item is None:
            raise HTTPException(404, "Không tìm thấy lần kiểm tra.")
        if options.target == "defect_type" and (item.get("defect_analysis") or {}).get("status") in (None, "unavailable"):
            raise HTTPException(422, "Lần kiểm tra này chưa có kết quả phân tích loại lỗi để đánh giá.")
        review = {**options.model_dump(), "note": options.note.strip(), "updated_at": datetime.now(timezone.utc).isoformat()}
        try:
            return history.save_review(item_id, review)
        except KeyError:
            raise HTTPException(404, "Lần kiểm tra đã được xóa.")

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
