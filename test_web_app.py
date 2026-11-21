"""Isolated HTTP contract tests; service doubles never become production models."""
import importlib
import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from ml.features import APP_DIR
from ml.defect_types import TYPE_CODES
from ml.hog_svm import MODEL_NAMES, HogSvmService as ModelService
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

    def predict(self, image, model="svm"):
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
        return self.client.post("/api/predict", files={"file": ("anh_chai.jpg", self.image_bytes, "image/jpeg")}, data={"model": "svm", **data})

    def test_home_static_and_single_image_contract(self):
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn("Kiểm tra chai nhựa", home.text)
        self.assertEqual(self.client.get("/static/svm.js").status_code, 200)
        self.assertEqual(self.client.get("/static/svm.css").status_code, 200)
        self.assertIn("HOG + SVM", home.text)
        self.assertNotIn("So sánh cả 3 mô hình", home.text)
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
            self.assertEqual(len(status["models"]), 1)
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
        response = self.client.post("/api/predict", files={"file": ("../../unsafe.jpg", self.image_bytes, "image/jpeg")}, data={"model": "svm"})
        self.assertEqual(response.status_code, 200)
        item = response.json()
        self.assertEqual(item["filename"], "unsafe.jpg")
        self.assertEqual(item["label"], "uncertain")
        self.assertEqual(self.history.get(item["id"])["label"], "uncertain")
        self.assertEqual(self.service.calls[0]["model"], "svm")
        self.assertTrue(self.history.image_path(item["id"]).is_file())





    def test_only_svm_is_available_and_comparison_route_is_removed(self):
        self.assertEqual([model["id"] for model in self.client.get("/api/models").json()["models"]], ["svm"])
        self.assertEqual(self.client.post("/api/compare").status_code, 404)
        for model in ("random_forest", "mobilenet"):
            with self.subTest(model=model):
                self.assertEqual(self.upload(model=model).status_code, 422)
        self.assertEqual(self.service.calls, [])
        self.assertEqual(self.history.list(), [])

    def test_history_csv_is_utf8_and_filename_is_not_an_excel_formula(self):
        response = self.client.post("/api/predict", files={"file": ("=1+1.jpg", self.image_bytes, "image/jpeg")}, data={"model": "svm"})
        self.assertEqual(response.status_code, 200)
        response = self.client.get("/api/history.csv")
        self.assertEqual(response.status_code, 200)
        import csv
        import io
        rows = list(csv.reader(io.StringIO(response.content.decode("utf-8-sig"))))
        self.assertEqual(rows[1][1], "'=1+1.jpg")
        self.assertEqual(rows[1][3], "Bình thường")
        self.assertIn("attachment", response.headers["content-disposition"])

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

    def test_library_groups_show_all_five_source_defects_and_normal_without_predicting(self):
        for code in TYPE_CODES[1:]:
            record_id = f"reference-{code}"
            path = self.root / f"{record_id}.jpg"
            path.write_bytes(self.image_bytes)
            self.manifest["records"].append({"id": record_id, "path": path.relative_to(APP_DIR).as_posix(),
                                             "split": "test", "label": "defect", "group": record_id, "defect_types": [code]})
        self.write_manifest()
        payload = self.client.get("/api/samples?split=test&limit=1").json()
        self.assertEqual([item["code"] for item in payload["categories"]], list(TYPE_CODES))
        for category in payload["categories"]:
            self.assertEqual(category["count"], 1)
            self.assertEqual(self.client.get(category["preview_url"]).status_code, 200)
            group = self.client.get(f"/api/samples?split=test&label={category['code']}").json()
            self.assertEqual(group["total"], 1)
            self.assertEqual(group["categories"], payload["categories"])
            record = group["samples"][0]
            if category["code"] == "normal":
                self.assertEqual(record["label"], "normal")
            else:
                self.assertIn(category["code"], record["defect_types"])
        self.assertEqual(self.service.calls, [])
        self.assertEqual(self.history.list(), [])

    def test_library_type_counts_follow_split_and_stay_constant_across_pages(self):
        for split, suffix in (("test", "a"), ("test", "b"), ("val", "c")):
            record_id = f"no-cap-{suffix}"
            self.manifest["records"].append({"id": record_id, "path": self.manifest["records"][0]["path"],
                                             "split": split, "label": "defect", "group": record_id,
                                             "defect_types": ["defect_no_cap"]})
        self.write_manifest()
        first = self.client.get("/api/samples?split=test&label=defect_no_cap&limit=1").json()
        second = self.client.get("/api/samples?split=test&label=defect_no_cap&limit=1&offset=1").json()
        self.assertEqual(first["total"], 2)
        self.assertEqual(first["categories"], second["categories"])
        self.assertNotEqual(first["samples"][0]["id"], second["samples"][0]["id"])
        validation = self.client.get("/api/samples?split=val&label=defect_no_cap").json()
        self.assertEqual(validation["total"], 1)
        empty = self.client.get("/api/samples?split=val&label=defect_label_damage").json()
        self.assertEqual(empty["samples"], [])
        category = next(item for item in empty["categories"] if item["code"] == "defect_label_damage")
        self.assertEqual(category["count"], 0)
        self.assertIsNone(category["preview_url"])
        self.assertEqual(self.client.get("/api/samples?label=defect_dent").status_code, 422)

    def test_train_options_validated_without_spawning_training(self):
        with patch.object(self.app.state.training, "start", return_value={"state": "running", "message": "isolated test"}) as start:
            for options in ({"seed": -1}, {"seed": 2**32}, {"epochs": 8}, {"batch_size": 8}):
                with self.subTest(options=options):
                    self.assertEqual(self.client.post("/api/train", json=options).status_code, 422)
            start.assert_not_called()
            response = self.client.post("/api/train", json={"seed": 42})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()["state"], "running")
            start.assert_called_once()
            self.assertEqual(start.call_args.args[0].seed, 42)

    def test_batch_retry_is_idempotent_and_rejects_different_content_or_model(self):
        data = {"batch_id": "a" * 32, "position": "0", "model": "svm"}
        def send(content=None, **changes):
            return self.client.post("/api/batch/predict", files={"file": ("batch.jpg", content or self.image_bytes, "image/jpeg")}, data={**data, **changes})
        first = send()
        self.assertEqual(first.status_code, 200, first.text)
        retry = send()
        self.assertEqual(first.json()["id"], retry.json()["id"])
        self.assertEqual(first.json()["batch_id"], "a" * 32)
        self.assertEqual(first.json()["batch_position"], 0)
        self.assertEqual(len(self.service.calls), 1)
        self.assertEqual(len(self.history.list()), 1)
        _, changed = cv2.imencode(".jpg", np.zeros_like(self.pixels))
        self.assertEqual(send(changed.tobytes()).status_code, 409)
        self.assertEqual(send(model="random_forest").status_code, 422)
        self.assertEqual(len(self.service.calls), 1)

    def test_batch_bad_image_does_not_prevent_other_positions_or_create_history(self):
        def send(position, content, **changes):
            return self.client.post("/api/batch/predict", files={"file": ("batch.jpg", content, "image/jpeg")}, data={"batch_id": "b"*32, "position": position, "model": "svm", **changes})
        self.assertEqual(send(0, self.image_bytes).status_code, 200)
        self.assertEqual(send(1, b"broken image").status_code, 422)
        self.assertEqual(send(2, self.image_bytes).status_code, 200)
        self.assertEqual({item["batch_position"] for item in self.history.list()}, {0, 2})
        for changes in ({"position": -1}, {"position": 50}, {"batch_id": "../invalid"}, {"model": "unknown"}):
            form = {"batch_id": "b"*32, "position": 3, "model": "svm", **changes}
            response = self.client.post("/api/batch/predict", files={"file": ("image.jpg", self.image_bytes, "image/jpeg")}, data=form)
            self.assertEqual(response.status_code, 422)
        self.assertEqual(len(self.history.list()), 2)

    def test_batch_processing_lock_blocks_concurrent_prediction_and_training(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        entered, release = Event(), Event()
        original = self.service.predict
        def wait_for_release(image, model):
            entered.set()
            if not release.wait(5): raise RuntimeError("test release timed out")
            return original(image, model)
        def request():
            return self.client.post("/api/batch/predict", files={"file": ("image.jpg", self.image_bytes, "image/jpeg")}, data={"batch_id": "c"*32, "position": 0, "model": "svm"})
        with patch.object(self.service, "predict", side_effect=wait_for_release), patch.object(self.app.state.training, "start") as start:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(request)
                try:
                    self.assertTrue(entered.wait(3))
                    self.assertEqual(self.upload(model="svm").status_code, 409)
                    self.assertEqual(self.client.post("/api/train", json={"seed": 42}).status_code, 409)
                    start.assert_not_called()
                finally:
                    release.set()
                self.assertEqual(future.result(timeout=5).status_code, 200)

    def test_review_is_separate_from_prediction_and_manifest_and_can_be_updated(self):
        item = self.upload(model="svm").json()
        manifest_bytes = self.manifest_path.read_bytes()
        response = self.client.put(f"/api/history/{item['id']}/review", json={"verdict": "incorrect", "corrected_label": "defect", "note": "  visible issue  "})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["note"], "visible issue")
        stored = self.client.get(f"/api/history/{item['id']}").json()
        self.assertEqual(stored["label"], item["label"])
        self.assertEqual(stored["confidence"], item["confidence"])
        self.assertEqual(stored["reviews"]["quality"]["corrected_label"], "defect")
        self.assertEqual(self.manifest_path.read_bytes(), manifest_bytes)
        self.assertEqual(len(self.service.calls), 1)
        response = self.client.put(f"/api/history/{item['id']}/review", json={"verdict": "unsure", "note": "Need another angle"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["corrected_label"])
        self.assertEqual(len(self.history.get(item["id"])["reviews"]), 1)

    def test_quality_and_type_reviews_are_independent_and_filter_without_duplicates(self):
        original = self.service.predict
        def with_types(image, model):
            return {**original(image, model), "defect_analysis": {"status": "typed", "code": "defect_no_cap", "name": "Thiếu nắp"}}
        with patch.object(self.service, "predict", side_effect=with_types): item = self.upload().json()
        other = self.upload().json()
        self.assertEqual(self.client.put(f"/api/history/{item['id']}/review", json={"verdict": "correct"}).status_code, 200)
        self.assertEqual(self.client.put(f"/api/history/{item['id']}/review", json={"target": "defect_type", "verdict": "incorrect", "corrected_type": "defect_cloudy", "note": "Human observation only"}).status_code, 200)
        self.assertEqual(set(self.history.get(item["id"])["reviews"]), {"quality", "defect_type"})
        for filter_name, expected in (("incorrect", [item["id"]]), ("correct", [item["id"]]), ("unreviewed", [other["id"]]), ("unsure", [])):
            rows = self.client.get(f"/api/history?review={filter_name}").json()["items"]
            self.assertEqual([row["id"] for row in rows], expected)
        self.assertEqual(len(self.client.get("/api/history").json()["items"]), 2)
        self.assertEqual(self.client.get("/api/history?review=invalid").status_code, 422)

    def test_invalid_reviews_are_rejected_and_deleted_items_have_no_reviews(self):
        item = self.upload().json()
        url = f"/api/history/{item['id']}/review"
        for body in ({"verdict": "invalid"}, {"verdict": "incorrect", "corrected_label": "other"},
                     {"verdict": "correct", "corrected_label": "defect"}, {"verdict": "unsure", "note": "x"*1001},
                     {"target": "quality", "verdict": "incorrect", "corrected_type": "defect_no_cap"},
                     {"target": "defect_type", "verdict": "incorrect", "corrected_label": "normal"},
                     {"target": "defect_type", "verdict": "correct"}):
            self.assertEqual(self.client.put(url, json=body).status_code, 422)
        self.assertNotIn("reviews", self.history.get(item["id"]))
        self.assertEqual(self.client.put("/api/history/missing/review", json={"verdict": "correct"}).status_code, 404)
        self.assertEqual(self.client.put(url, json={"verdict": "correct"}).status_code, 200)
        self.assertEqual(self.client.delete(f"/api/history/{item['id']}").status_code, 200)
        with closing(self.history.connect()) as con: self.assertEqual(con.execute("SELECT count(*) FROM reviews").fetchone()[0], 0)
        self.assertEqual(self.client.get(f"/api/history/{item['id']}").status_code, 404)

    def test_csv_escapes_user_review_notes(self):
        import csv, io
        item = self.upload().json()
        self.client.put(f"/api/history/{item['id']}/review", json={"verdict": "incorrect", "note": "=2+2"})
        rows = list(csv.reader(io.StringIO(self.client.get("/api/history.csv").content.decode("utf-8-sig"))))
        self.assertEqual(rows[1][9], "incorrect")
        self.assertEqual(rows[1][11], "'=2+2")

    def test_old_history_database_is_preserved_when_new_tables_are_added(self):
        import sqlite3
        folder = self.root / "legacy-history"
        folder.mkdir()
        item = {"id": "old", "created_at": "2026-01-01T00:00:00Z", "filename": "old.jpg", "label": "normal"}
        with closing(sqlite3.connect(folder/"history.db")) as con, con:
            con.execute("CREATE TABLE predictions(id TEXT PRIMARY KEY,created_at TEXT NOT NULL,filename TEXT NOT NULL,payload TEXT NOT NULL)")
            con.execute("INSERT INTO predictions VALUES(?,?,?,?)", (item["id"],item["created_at"],item["filename"],json.dumps(item)))
        upgraded = HistoryStore(folder)
        self.assertEqual(upgraded.get("old"), item)
        self.assertEqual(upgraded.list(review="unreviewed"), [item])

    def test_web_training_requires_matched_crops_and_uses_owned_artifact_paths(self):
        response = self.client.post("/api/train", json={"seed": 42})
        self.assertEqual(response.status_code, 409)
        self.manifest["input_mode"] = "auto_bottle_roi"
        self.write_manifest()
        with patch("web_training.subprocess.Popen") as popen:
            popen.return_value.poll.return_value = None
            response = self.client.post("/api/train", json={"seed": 42})
            self.assertEqual(response.status_code, 202)
            command = popen.call_args.args[0]
            self.assertTrue(command[2].endswith("train_hog_svm.py"))
            self.assertNotIn("--epochs", command)
            self.assertEqual(command[command.index("--seed") + 1], "42")
            self.assertEqual(command[command.index("--manifest") + 1], str(self.manifest_path))
            self.assertEqual(command[command.index("--artifact-root") + 1], str(self.service.artifact_root))
            self.assertEqual(self.client.post("/api/train", json={"seed": 42}).status_code, 409)
            popen.assert_called_once()
            popen.return_value.poll.return_value = 1
            status = self.client.get("/api/status").json()
            self.assertEqual(status["training"]["state"], "failed")
            self.assertFalse(status["training"]["running"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
