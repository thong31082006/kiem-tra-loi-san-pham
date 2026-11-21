"""Local web application: trained model inference from one image, with no reference input."""
import json
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

from ml.hog_svm import HogSvmService
from web_dataset import DatasetCatalog, InvalidSampleFilter
from web_history import HistoryStore
from web_training import TrainingManager, TrainingOptions

APP_DIR = Path(__file__).resolve().parent
MANIFEST = APP_DIR / "data" / "training" / "manifest.json"
SCOPE = (
    "Mô hình thử nghiệm cho mẫu chai trong bộ dữ liệu Ron 88. "
    "HOG + SVM kiểm tra một chai đứng, thấy rõ nắp và đáy. "
    "Chưa kiểm chứng chai thật nằm ngang, góc nhìn mới, mọi loại chai hoặc lỗi bị che khuất."
)
MAX_UPLOAD = 15 * 1024 * 1024


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


def create_app(service=None, history=None, manifest_path=None):
    application = FastAPI(title="Kiểm tra chai nhựa — Nhóm 8", version="2.0")
    service = service or HogSvmService()
    history = history or HistoryStore()
    manifest_path = Path(manifest_path or MANIFEST)
    catalog = DatasetCatalog(manifest_path)
    manager = TrainingManager(service.artifact_root, manifest_path, catalog=catalog)
    prediction_lock = threading.Lock()
    application.state.model_service = service
    application.state.history = history
    application.state.training = manager
    application.state.dataset_catalog = catalog
    application.mount("/static", StaticFiles(directory=APP_DIR / "web"), name="static")

    def status_data():
        status = service.status()
        dataset = catalog.summary(SCOPE)
        return {**status, "dataset": dataset, "training": manager.status(), "scope": SCOPE}

    @application.get("/", include_in_schema=False)
    def homepage():
        return FileResponse(APP_DIR / "web" / "svm.html")

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

    def predict_sync(image):
        if not prediction_lock.acquire(blocking=False):
            raise HTTPException(409, "Đang kiểm tra ảnh khác. Hãy thử lại sau giây lát.")
        try:
            return service.predict(image, model="svm")
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
    async def predict(file: UploadFile = File(...), model: str = Form("svm")):
        if model != "svm":
            raise HTTPException(422, "Mô hình không hợp lệ.")
        image = await decode_upload(file)
        try:
            result = await run_in_threadpool(predict_sync, image)
        except FileNotFoundError as error:
            raise HTTPException(409, str(error))
        except (ValueError, RuntimeError) as error:
            raise HTTPException(422, str(error))
        item = await run_in_threadpool(save_prediction, image, result, file.filename)
        return item

    @application.post("/api/batch/predict")
    async def batch_predict(file: UploadFile = File(...), model: str = Form("svm"),
                            batch_id: str = Form(..., pattern=r"^[a-f0-9]{32}$"),
                            position: int = Form(..., ge=0, le=49)):
        if model != "svm":
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
                return item
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
            # All text columns are escaped together before writer.writerow below.
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
        try:
            return catalog.browse(split, label, limit, offset)
        except InvalidSampleFilter as error:
            raise HTTPException(422, str(error)) from error

    @application.get("/api/samples/{sample_id}/image")
    def sample_image(sample_id: str):
        row = catalog.find(sample_id)
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
