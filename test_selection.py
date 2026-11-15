"""Check abstention-aware model selection, without fitting on Test images."""
import unittest

from ml.selection import accept_candidate, choose_pipeline_threshold, pipeline_metrics


class SelectionTests(unittest.TestCase):
    def test_abstentions_count_against_both_class_recalls(self):
        summary = pipeline_metrics([0, 0, 1, 1], [0.1, 0.45, 0.55, 0.9], 0.5)
        self.assertEqual(summary["confusion_matrix"], [[1, 0, 1], [0, 1, 1]])
        self.assertEqual(summary["balanced_correct"], 0.5)
        self.assertEqual(summary["coverage"], 0.5)

    def test_threshold_does_not_turn_confident_normal_into_uncertain(self):
        threshold, summary = choose_pipeline_threshold([0, 0, 1, 1], [0.2, 0.3, 0.7, 0.9])
        self.assertAlmostEqual(threshold, 0.5)
        self.assertEqual(summary["confusion_matrix"], [[2, 0, 0], [0, 2, 0]])

    def test_false_alarm_reduction_alone_is_not_enough(self):
        baseline = {"balanced_correct": 0.9, "false_positives": 2, "recall": 0.9}
        self.assertFalse(accept_candidate({"balanced_correct": 0.91, "false_positives": 0, "recall": 0.85}, baseline))
        self.assertFalse(accept_candidate({"balanced_correct": 0.95, "false_positives": 3, "recall": 1.0}, baseline))
        self.assertFalse(accept_candidate({"balanced_correct": 0.9, "false_positives": 1, "recall": 0.9}, baseline))
        self.assertTrue(accept_candidate({"balanced_correct": 0.92, "false_positives": 1, "recall": 0.9}, baseline))


if __name__ == "__main__":
    unittest.main(verbosity=2)
