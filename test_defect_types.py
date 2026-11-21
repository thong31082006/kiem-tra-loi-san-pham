"""Contracts for naming faults from scores, independent of filenames/reference labels."""
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from ml.defect_types import TYPE_CODES, classify_type_scores
from ml.hog_svm import HogSvmService


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

    def test_quality_and_type_use_one_svm_prediction_on_pixel_features(self):
        pixels = np.full((128, 64, 3), 170, np.uint8)
        service = HogSvmService()
        service._refresh = Mock(return_value=Path("."))
        service._report = {"models": {"svm": {"decision_threshold": 0.5, "uncertainty_threshold": 0.65}},
                           "policies": self.policies()}
        service._loaded = Mock()
        service._loaded.predict_proba.return_value = np.array([[0.02, 0.02, 0.9, 0.02, 0.02, 0.02]])
        service._locator = Mock()
        service._locator.locate.return_value = (pixels, {})
        result = service.predict(pixels)
        self.assertEqual(result["label"], "defect")
        self.assertEqual(result["defect_analysis"]["code"], "defect_no_cap")
        self.assertEqual(result["defect_analysis"]["agreement"], "agree")
        service._loaded.predict_proba.assert_called_once()
        self.assertEqual(service._loaded.predict_proba.call_args.args[0].shape, (1, 3780))

    def test_disabled_type_does_not_hide_quality_defect(self):
        policies = self.policies()
        policies["defect_label_damage"]["enabled"] = False
        service = HogSvmService()
        service._refresh = Mock(return_value=Path("."))
        service._report = {"models": {"svm": {"decision_threshold": 0.5, "uncertainty_threshold": 0.65}},
                           "policies": policies}
        service._loaded = Mock()
        service._loaded.predict_proba.return_value = np.array([[0.02, 0.02, 0.02, 0.02, 0.02, 0.9]])
        service._locator = Mock()
        service._locator.locate.return_value = (np.zeros((128, 64, 3), np.uint8), {})
        result = service.predict(np.zeros((128, 64, 3), np.uint8))
        self.assertEqual(result["label"], "defect")
        self.assertEqual(result["defect_analysis"]["status"], "uncertain")


if __name__ == "__main__":
    unittest.main(verbosity=2)
