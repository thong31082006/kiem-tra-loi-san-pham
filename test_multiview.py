"""Real-capture labels and split isolation, without modifying the source dataset."""
import tempfile
import csv
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from ml.features import APP_DIR, read_bgr
from scripts.import_multiview import import_collection, validate_rows, visible_label


class MultiViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        rng = np.random.default_rng(3)
        for index in range(6):
            cv2.imwrite(str(self.folder / f"{index}.png"), rng.integers(0, 255, (64, 96, 3), dtype=np.uint8))

    def row(self, index, bottle="bottle1", session="session1", **kwargs):
        return {"path": f"{index}.png", "bottle_id": bottle, "session_id": session, "view": "front",
                "defect_type": "normal", "defect_visible": "false", "split": "", **kwargs}

    def test_hidden_defect_is_uncertain_and_keeps_the_physical_defect_type(self):
        self.assertEqual(visible_label("defect_no_cap", "false"), ("uncertain", ["defect_no_cap"]))
        self.assertEqual(visible_label("defect_no_cap", "true"), ("defect", ["defect_no_cap"]))
        self.assertEqual(visible_label("normal", "false"), ("normal", []))
        with self.assertRaises(ValueError):
            visible_label("normal", "true")

    def test_transitive_bottle_session_links_stay_together(self):
        rows = [self.row(0, "A", "S1"), self.row(1, "A", "S2"), self.row(2, "B", "S2"),
                self.row(3, "C", "S3"), self.row(4, "D", "S4")]
        samples = validate_rows(rows, self.folder)
        self.assertEqual(len({r["group"] for r in samples[:3]}), 1)
        self.assertEqual(len({r["split"] for r in samples[:3]}), 1)
        self.assertEqual({r["split"] for r in samples}, {"train", "val", "test"})

    def test_conflicting_splits_for_same_bottle_or_session_are_rejected(self):
        for rows in ([self.row(0, split="train"), self.row(1, session="other", split="test")],
                     [self.row(0, bottle="A", split="train"), self.row(1, bottle="B", split="val")]):
            with self.assertRaisesRegex(ValueError, "rò rỉ"):
                validate_rows(rows, self.folder)

    def test_duplicate_pixels_are_rejected_even_with_different_names(self):
        (self.folder / "copy.png").write_bytes((self.folder / "0.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "trùng"):
            validate_rows([self.row(0), self.row(1, path="copy.png")], self.folder)

    def test_paths_cannot_escape_selected_image_directory(self):
        with self.assertRaises(ValueError):
            validate_rows([self.row(0, path="../outside.png")], self.folder)

    def test_empty_csv_and_unknown_metadata_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows([], self.folder)
        with self.assertRaises(ValueError):
            validate_rows([self.row(0, view="any-angle")], self.folder)
        with self.assertRaises(ValueError):
            validate_rows([self.row(0, defect_visible="maybe")], self.folder)

    def test_real_import_preserves_pixels_and_stages_hidden_defects_without_training_records(self):
        rows = [self.row(0), self.row(1, view="back", defect_type="defect_label_damage", defect_visible="false")]
        csv_path = self.folder / "captures.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with tempfile.TemporaryDirectory(prefix=".multiview-test-", dir=APP_DIR / "data" / "collection") as destination:
            import_collection(csv_path, self.folder, Path(destination))
            result = json.loads((Path(destination) / "collection.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "awaiting_review")
            self.assertNotIn("records", result)
            self.assertEqual(result["samples"][1]["label"], "uncertain")
            for index, sample in enumerate(result["samples"]):
                np.testing.assert_array_equal(read_bgr(APP_DIR / sample["path"]), read_bgr(self.folder / f"{index}.png"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
