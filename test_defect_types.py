"""Contracts for naming faults from scores, independent of filenames/reference labels."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from ml.defect_types import DefectTypeService, TYPE_CODES, classify_type_scores
from ml.inference import ModelService


class BinaryFixture:
    classes_ = np.array([0, 1])
    def __init__(self, defect_score): self.defect_score = defect_score
    def predict_proba(self, features): return np.array([[1 - self.defect_score, self.defect_score]])


class DefectTypeTests(unittest.TestCase):
    def policies(self):
        return {code: {"enabled": True, "threshold": 0.65} for code in TYPE_CODES}

    def test_name_requires_validated_threshold(self):
        scores = [0.05, 0.8, 0.05, 0.04, 0.03, 0.03]
        result = classify_type_scores(scores, self.policies())
        self.assertEqual(result["status"], "typed")
        self.assertEqual(result["code"], "defect_low_fill")
        self.assertEqual(result["name"], "Thiếu mức nước")
        weak = classify_type_scores([0.1, 0.3, 0.2, 0.15, 0.15, 0.1], self.policies())
        self.assertEqual(weak["status"], "uncertain")
        self.assertIsNone(weak["code"])

    def test_disabled_type_is_not_substituted_with_another_fault(self):
        policies = self.policies()
        policies["defect_low_fill"]["enabled"] = False
        result = classify_type_scores([0.01, 0.95, 0.01, 0.01, 0.01, 0.01], policies)
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["code"])

    def test_normal_and_unsupported_capabilities_are_explicit(self):
        result = classify_type_scores([0.9, 0.02, 0.02, 0.02, 0.02, 0.02], self.policies())
        self.assertEqual(result["status"], "normal")
        self.assertEqual(result["code"], "normal")
        self.assertNotIn("defect_dent", result["scores"])
        self.assertNotIn("defect_cloudy", result["scores"])
        self.assertEqual(len(result["unsupported"]), 2)

    def test_invalid_scores_cannot_generate_fault_names(self):
        for scores in ([0.8, 0.2], [1, 1, 1, 1, 1, 1], [float("nan"), 0, 0, 0, 0, 1], [-0.1, 1.1, 0, 0, 0, 0]):
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError): classify_type_scores(scores, self.policies())

    def test_missing_or_escaping_type_artifact_never_claims_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            service = DefectTypeService(folder)
            self.assertFalse(service.status()["ready"])
            self.assertEqual(service.predict(np.zeros((64, 64, 3), np.uint8))["status"], "unavailable")
            (Path(folder) / "latest.json").write_text(json.dumps({"run_id": "../outside"}), encoding="utf-8")
            self.assertFalse(service.status()["ready"])

    def test_type_analysis_preserves_binary_result_and_only_receives_pixels(self):
        pixels = np.full((128, 64, 3), 170, np.uint8)
        service = ModelService(enable_defect_types=False)
        service._refresh = Mock(return_value=Path("."))
        service._report = {"models": {"svm": {"decision_threshold": 0.5, "uncertainty_threshold": 0.65}}}
        service._load = Mock(return_value=BinaryFixture(0.05))
        baseline = service.predict(pixels, "svm")
        service._type_service = Mock()
        service._type_service.predict.return_value = {"status": "typed", "code": "defect_no_cap", "name": "Thiếu nắp", "score": 0.9}
        with_types = service.predict(pixels, "svm")
        self.assertEqual(with_types["label"], baseline["label"])
        self.assertEqual(with_types["probabilities"], baseline["probabilities"])
        self.assertEqual(with_types["defect_analysis"]["agreement"], "review")
        args, kwargs = service._type_service.predict.call_args
        self.assertEqual(kwargs, {})
        self.assertEqual(len(args), 1)
        np.testing.assert_array_equal(args[0], pixels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
