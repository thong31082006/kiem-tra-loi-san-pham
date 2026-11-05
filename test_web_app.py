"""Isolated HTTP contract tests; service doubles never become production models."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from ml.features import APP_DIR
from ml.inference import MODEL_NAMES, ModelService
from web_history import HistoryStore


class StubModelService:
    """Exercise API plumbing only; does not claim trained classification accuracy."""
    def __init__(self, artifact_root, ready=True, label="normal", confidence=0.8):
        self.artifact_root = Path(artifact_root)
        self.ready, self.label, self.confidence = ready, label, confidence
        self.calls = []
        self._report = None

    def status(self):
        return {
            "ready": self.ready,
            "models": [{"id": key, "name": name, "ready": self.ready, "metrics": {}} for key, name in MODEL_NAMES.items()],
        }

    def predict(self, image, model="mobilenet"):
        if not self.ready:
            raise FileNotFoundError("Chưa có mô hình đã huấn luyện.")
        self.calls.append({"shape": image.shape, "image": image.copy(), "model": model})
        return {
            "label": self.label, "raw_label": "normal", "confidence": self.confidence,
            "probabilities": {"normal": 0.8, "defect": 0.2}, "model": model,
            "model_name": MODEL_NAMES[model], "latency_ms": 1.25,
            "run_id": "isolated-test-double", "cold_start": False,
        }


class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # web_api exposes an app at import time; redirect that constructor too so
        # importing the module cannot initialize the user's actual history DB.
        cls.import_temp = tempfile.TemporaryDirectory(prefix="bottle-web-import-")
        with patch("web_history.HistoryStore", side_effect=lambda *args, **kwargs: HistoryStore(Path(cls.import_temp.name) / "history")):
            cls.api = importlib.import_module("web_api")
        cls.api.HistoryStore = HistoryStore

    @classmethod
    def tearDownClass(cls):
        cls.import_temp.cleanup()

    def setUp(self):
        dataset_root = APP_DIR / "data" / "datasets"
        dataset_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix=".web-api-test-", dir=dataset_root)
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.history = HistoryStore(self.root / "history")
        self.service = StubModelService(self.root / "artifacts")
        self.pixels = np.full((96, 64, 3), 236, dtype=np.uint8)
        cv2.rectangle(self.pixels, (22, 20), (42, 84), (170, 120, 80), -1)
        ok, encoded = cv2.imencode(".jpg", self.pixels)
        self.assertTrue(ok)
        self.image_bytes = encoded.tobytes()
        records = []
        for image_id, split, label in (("train-normal", "train", "normal"), ("val-defect", "val", "defect"), ("test-normal", "test", "normal"), ("test-defect", "test", "defect")):
            path = self.root / f"{image_id}.jpg"
            path.write_bytes(self.image_bytes)
            records.append({"id": image_id, "path": path.relative_to(APP_DIR).as_posix(), "label": label, "split": split, "group": image_id, "defect_types": [] if label == "normal" else ["test-fixture"]})
        self.manifest = {"records": records, "source_url": "isolated-test-fixture", "split_note": "HTTP test only"}
        self.manifest_path = self.root / "manifest.json"
        self.write_manifest()
        self.app = self.api.create_app(self.service, self.history, self.manifest_path)
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest, ensure_ascii=False), encoding="utf-8")

    def upload(self, **data):
        return self.client.post("/api/predict", files={"file": ("anh_chai.jpg", self.image_bytes, "image/jpeg")}, data={"model": "mobilenet", **data})

    def test_home_static_and_single_image_contract(self):
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn("Kiểm tra chai nhựa", home.text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/static/app.css").status_code, 200)
        schema = self.client.get("/openapi.json").json()
        reference = schema["paths"]["/api/predict"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
        body = schema["components"]["schemas"][reference.rsplit("/", 1)[-1]]
        self.assertEqual(set(body["properties"]), {"file", "model"})
        self.assertEqual(body["required"], ["file"])

    def test_untrained_models_have_no_invented_metrics(self):
        service = ModelService(self.root / "no-trained-artifacts")
        with TestClient(self.api.create_app(service, self.history, self.manifest_path)) as client:
            status = client.get("/api/status").json()
            self.assertFalse(status["ready"])
            self.assertEqual(len(status["models"]), 3)
            self.assertTrue(all(not model["ready"] and model["metrics"] == {} for model in status["models"]))
            response = client.get("/api/models").json()
            self.assertEqual(response["report"], {})
            self.assertEqual(response["dataset"]["image_count"], 4)
            self.assertEqual(client.get("/api/report.json").status_code, 409)

    def test_prediction_saved_image_retrieved_and_deleted(self):
        response = self.upload(model="svm")
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()
        self.assertEqual(item["label"], "normal")
        self.assertEqual(self.service.calls[0]["model"], "svm")
        self.assertEqual(self.service.calls[0]["shape"], (96, 64, 3))
        self.assertEqual(self.history.get(item["id"]), item)
        history = self.client.get("/api/history").json()["items"]
        self.assertEqual([row["id"] for row in history], [item["id"]])
        image = self.client.get(item["image_url"])
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.headers["content-type"], "image/jpeg")
        self.assertEqual(cv2.imdecode(np.frombuffer(image.content, dtype=np.uint8), cv2.IMREAD_COLOR).shape, (96, 64, 3))
        self.assertEqual(self.client.delete(f"/api/history/{item['id']}").status_code, 200)
        self.assertEqual(self.client.get(item["image_url"]).status_code, 404)
        self.assertEqual(self.client.delete(f"/api/history/{item['id']}").status_code, 404)
        self.assertEqual(self.history.list(), [])
        self.assertFalse(self.history.image_path(item["id"]).exists())

    def test_unready_prediction_does_not_create_history(self):
        self.service.ready = False
        response = self.upload()
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.history.list(), [])
        self.assertEqual(list(self.history.upload_dir.iterdir()), [])

    def test_invalid_empty_small_oversized_uploads_and_invalid_model(self):
        for content, status in ((b"", 413), (b"not an image", 422)):
            with self.subTest(content=content):
                response = self.client.post("/api/predict", files={"file": ("bad.jpg", content, "image/jpeg")})
                self.assertEqual(response.status_code, status)
        _, tiny = cv2.imencode(".png", np.zeros((24, 24, 3), dtype=np.uint8))
        response = self.client.post("/api/predict", files={"file": ("tiny.png", tiny.tobytes(), "image/png")})
        self.assertEqual(response.status_code, 422)
        _, encoded_png = cv2.imencode(".png", self.pixels)
        broken_png = bytearray(encoded_png.tobytes())
        position = 8
        while position < len(broken_png):
            length = int.from_bytes(broken_png[position:position + 4], "big")
            if broken_png[position + 4:position + 8] == b"IDAT":
                broken_png[position + 8 + length] ^= 1  # Corrupt the chunk CRC, not the image header.
                break
            position += length + 12
        response = self.client.post("/api/predict", files={"file": ("bad-crc.png", bytes(broken_png), "image/png")})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.api.MAX_UPLOAD, 15 * 1024 * 1024)
        # Exercise the MAX+1 streaming boundary without allocating a 15 MB fixture.
        with patch.object(self.api, "MAX_UPLOAD", 128):
            response = self.client.post("/api/predict", files={"file": ("large.jpg", bytes(129), "image/jpeg")})
            self.assertEqual(response.status_code, 413)
        self.assertEqual(self.upload(model="unknown").status_code, 422)
        self.assertEqual(self.client.post("/api/predict", data={"model": "svm"}).status_code, 422)
        self.assertEqual(self.history.list(), [])
        self.assertEqual(self.service.calls, [])

    def test_uncertain_label_preserved_and_filename_sanitized(self):
        self.service.label, self.service.confidence = "uncertain", 0.55
        response = self.client.post("/api/predict", files={"file": ("../../unsafe.jpg", self.image_bytes, "image/jpeg")}, data={"model": "random_forest"})
        self.assertEqual(response.status_code, 200)
        item = response.json()
        self.assertEqual(item["filename"], "unsafe.jpg")
        self.assertEqual(item["label"], "uncertain")
        self.assertEqual(self.history.get(item["id"])["label"], "uncertain")
        self.assertEqual(self.service.calls[0]["model"], "random_forest")
        self.assertTrue(self.history.image_path(item["id"]).is_file())

    def test_sample_filters_pagination_limit_and_path_guard(self):
        response = self.client.get("/api/samples?split=test&label=defect&limit=1500")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 1)
        self.assertEqual([row["id"] for row in payload["samples"]], ["test-defect"])
        self.assertEqual(self.client.get(payload["samples"][0]["image_url"]).status_code, 200)
        page = self.client.get("/api/samples?split=test&limit=1&offset=1").json()
        self.assertEqual(page["total"], 2)
        self.assertEqual([row["id"] for row in page["samples"]], ["test-defect"])
        for query in ("limit=1501", "limit=0", "offset=-1", "split=invalid", "label=invalid"):
            with self.subTest(query=query):
                self.assertEqual(self.client.get(f"/api/samples?{query}").status_code, 422)
        self.assertEqual(self.client.get("/api/samples/missing/image").status_code, 404)
        self.manifest["records"].append({"id": "outside-dataset", "path": "web/index.html", "label": "normal", "split": "test", "group": "outside"})
        self.manifest["records"].append({"id": "file-removed", "path": (self.root / "file-removed.jpg").relative_to(APP_DIR).as_posix(), "label": "normal", "split": "test", "group": "missing-file"})
        self.write_manifest()
        self.assertEqual(self.client.get("/api/samples/outside-dataset/image").status_code, 404)
        self.assertEqual(self.client.get("/api/samples/file-removed/image").status_code, 404)

    def test_train_options_validated_without_spawning_training(self):
        with patch.object(self.app.state.training, "start", return_value={"state": "running", "message": "isolated test"}) as start:
            for options in ({"epochs": 0}, {"epochs": 31}, {"batch_size": 0}, {"batch_size": 17}):
                with self.subTest(options=options):
                    self.assertEqual(self.client.post("/api/train", json=options).status_code, 422)
            start.assert_not_called()
            response = self.client.post("/api/train", json={"epochs": 8, "batch_size": 8})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()["state"], "running")
            start.assert_called_once()
            self.assertEqual(start.call_args.args[0].epochs, 8)
            self.assertEqual(start.call_args.args[0].batch_size, 8)

    def test_web_training_requires_matched_crops_and_uses_owned_artifact_paths(self):
        response = self.client.post("/api/train", json={"epochs": 8, "batch_size": 8})
        self.assertEqual(response.status_code, 409)
        self.manifest["input_mode"] = "auto_bottle_roi"
        self.write_manifest()
        with patch.object(self.api.subprocess, "Popen") as popen:
            popen.return_value.poll.return_value = None
            response = self.client.post("/api/train", json={"epochs": 8, "batch_size": 8})
            self.assertEqual(response.status_code, 202)
            command = popen.call_args.args[0]
            self.assertTrue(command[2].endswith("train_and_evaluate.py"))
            self.assertEqual(command[command.index("--manifest") + 1], str(self.manifest_path))
            self.assertEqual(command[command.index("--artifact-root") + 1], str(self.service.artifact_root))
            self.assertEqual(self.client.post("/api/train", json={"epochs": 8}).status_code, 409)
            popen.assert_called_once()
            popen.return_value.poll.return_value = 1
            status = self.client.get("/api/status").json()
            self.assertEqual(status["training"]["state"], "failed")
            self.assertFalse(status["training"]["running"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
