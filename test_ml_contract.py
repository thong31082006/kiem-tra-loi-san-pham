"""Feature/manifest/artifact contracts without fitting or publishing any models."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ml.features import APP_DIR, CLASS_NAMES, as_rgb_pil, extract_features, read_bgr, read_model_image, validate_image
from ml.inference import ModelService
from ml.localization import BottleLocator
from ml.train import load_manifest


class MLContractTests(unittest.TestCase):
    def setUp(self):
        root = APP_DIR / "data" / "datasets"
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix=".ml-contract-test-", dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pixels = np.full((128, 64, 3), (230, 220, 210), dtype=np.uint8)
        cv2.rectangle(self.pixels, (20, 15), (44, 117), (130, 90, 60), -1)
        self.records = []
        for split, count in (("train", 2), ("val", 1), ("test", 1)):
            for label in CLASS_NAMES:
                for index in range(count):
                    record_id = f"{split}-{label}-{index}"
                    path = self.root / f"{record_id}.png"
                    ok, encoded = cv2.imencode(".png", self.pixels)
                    self.assertTrue(ok)
                    encoded.tofile(str(path))
                    self.records.append({"id": record_id, "path": path.relative_to(APP_DIR).as_posix(), "label": label, "split": split, "group": record_id})
        self.manifest = self.root / "manifest.json"

    def write_manifest(self):
        self.manifest.write_text(json.dumps({"records": self.records}), encoding="utf-8")

    def test_manifest_requires_both_classes_in_each_grouped_split(self):
        self.write_manifest()
        _, records, counts = load_manifest(self.manifest)
        self.assertEqual(len(records), 8)
        self.assertEqual(counts["train"], {"normal": 2, "defect": 2})
        self.assertEqual(counts["val"], {"normal": 1, "defect": 1})
        self.assertEqual(counts["test"], {"normal": 1, "defect": 1})

    def test_manifest_rejects_group_leakage_between_splits(self):
        self.records[-1]["group"] = self.records[0]["group"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Group"):
            load_manifest(self.manifest)

    def test_manifest_rejects_repeated_image_paths(self):
        self.records[-1]["path"] = self.records[0]["path"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "lặp"):
            load_manifest(self.manifest)

    def test_manifest_rejects_missing_class_and_invalid_label(self):
        original = [dict(record) for record in self.records]
        self.records = [record for record in self.records if not (record["split"] == "test" and record["label"] == "normal")]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Tập test"):
            load_manifest(self.manifest)
        self.records = original
        self.records[0]["label"] = "unknown"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Nhãn"):
            load_manifest(self.manifest)

    def test_pixel_features_match_after_unicode_file_decode(self):
        path = self.root / "ảnh_chai_thử.png"
        ok, encoded = cv2.imencode(".png", self.pixels)
        self.assertTrue(ok)
        encoded.tofile(str(path))
        decoded = read_bgr(path)
        np.testing.assert_array_equal(decoded, self.pixels)
        original = extract_features(self.pixels)
        np.testing.assert_array_equal(original, extract_features(decoded))
        self.assertEqual(original.shape, (3828,))
        self.assertEqual(original.dtype, np.float32)
        self.assertTrue(np.isfinite(original).all())
        np.testing.assert_array_equal(np.asarray(as_rgb_pil(decoded)), cv2.cvtColor(self.pixels, cv2.COLOR_BGR2RGB))

    def test_image_validation_rejects_bad_shape_dtype_and_empty(self):
        for image in (None, np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.float32), np.zeros((0, 0, 3), dtype=np.uint8)):
            with self.subTest(shape=getattr(image, "shape", None)):
                with self.assertRaises(ValueError):
                    validate_image(image)

    def test_training_reads_exact_automatic_crop_not_annotation_crop(self):
        crop = np.full((64, 28, 3), (17, 89, 204), np.uint8)
        cv2.rectangle(crop, (5, 7), (19, 49), (230, 180, 20), -1)
        path = self.root / "automatic_crop.png"
        ok, encoded = cv2.imencode(".png", crop)
        self.assertTrue(ok)
        encoded.tofile(str(path))
        row = {**self.records[0], "training_path": path.relative_to(APP_DIR).as_posix(), "bbox": [0, 0, -1, -1]}
        decoded = read_model_image(row)
        np.testing.assert_array_equal(decoded, crop)
        np.testing.assert_array_equal(extract_features(decoded), extract_features(crop))

    def test_manifest_rejects_missing_automatic_training_crop(self):
        self.records[0]["training_path"] = (self.root / "missing_crop.png").relative_to(APP_DIR).as_posix()
        self.write_manifest()
        with self.assertRaises(FileNotFoundError):
            load_manifest(self.manifest)
        self.records[0].pop("training_path")
        self.manifest.write_text(json.dumps({"input_mode": "auto_bottle_roi", "records": self.records}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "chuẩn bị"):
            load_manifest(self.manifest)

    def test_nested_neck_detection_does_not_become_second_bottle(self):
        full = (0.669, [211, 63, 365, 490])
        neck = (0.361, [207, 64, 349, 261])
        separate = (0.8, [500, 70, 650, 490])
        self.assertEqual(BottleLocator.suppress_nested([neck, full]), [full])
        self.assertEqual(len(BottleLocator.suppress_nested([neck, full, separate])), 2)

    def test_service_without_artifacts_has_no_ready_models_or_metrics(self):
        service = ModelService(self.root / "untrained")
        status = service.status()
        self.assertFalse(status["ready"])
        self.assertEqual(len(status["models"]), 3)
        self.assertTrue(all(not model["ready"] and model["metrics"] == {} for model in status["models"]))
        with self.assertRaises(FileNotFoundError):
            service.predict(self.pixels)

    def test_corrupt_or_escaping_artifact_pointer_is_rejected(self):
        artifacts = self.root / "artifacts"
        artifacts.mkdir()
        pointer = artifacts / "latest.json"
        pointer.write_text("{not valid json", encoding="utf-8")
        status = ModelService(artifacts).status()
        self.assertFalse(status["ready"])
        self.assertIn("error", status)
        pointer.write_text(json.dumps({"run_id": "../outside-artifacts"}), encoding="utf-8")
        status = ModelService(artifacts).status()
        self.assertFalse(status["ready"])
        self.assertIn("error", status)


if __name__ == "__main__":
    unittest.main(verbosity=2)
