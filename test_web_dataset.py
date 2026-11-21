"""Cache regressions: concurrent readers, atomic replacement and malformed metadata."""
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from web_dataset import DatasetCatalog


class DatasetCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="svm-catalog-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "manifest.json"
        self.catalog = DatasetCatalog(self.path)

    def write(self, record_id="before-1", destination=None):
        data = {"input_mode": "auto_bottle_roi", "records": [{"id": record_id, "split": "test",
                "label": "normal", "path": "data/datasets/source.jpg", "defect_types": []}]}
        (destination or self.path).write_text(json.dumps(data), encoding="utf-8")

    def test_concurrent_routes_share_one_parse_of_unchanged_manifest(self):
        self.write()
        reads = []
        original = Path.read_text

        def read(path, *args, **kwargs):
            if path == self.path:
                reads.append(path)
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read), ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.catalog.browse("test", "normal"), range(24)))
            self.assertEqual(self.catalog.summary("scope")["test"], 1)
            self.assertEqual(self.catalog.find("before-1")["label"], "normal")
            self.assertEqual(self.catalog.input_mode, "auto_bottle_roi")
        self.assertEqual(len(reads), 1)
        self.assertTrue(all(result["total"] == 1 for result in results))

    def test_atomic_replacement_with_same_size_and_mtime_invalidates_id_index(self):
        self.write()
        original = self.path.stat()
        self.assertIsNotNone(self.catalog.find("before-1"))
        replacement = self.path.with_name("replacement.json")
        self.write("after--1", replacement)
        self.assertEqual(replacement.stat().st_size, original.st_size)
        os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
        replacement.replace(self.path)
        self.assertIsNone(self.catalog.find("before-1"))
        self.assertEqual(self.catalog.browse()["samples"][0]["id"], "after--1")

    def test_missing_created_and_deleted_manifest_never_leaves_stale_gallery(self):
        self.assertEqual(self.catalog.summary("scope")["image_count"], 0)
        self.assertEqual(self.catalog.browse()["total"], 0)
        self.write()
        self.assertEqual(self.catalog.browse()["total"], 1)
        self.path.unlink()
        self.assertEqual(self.catalog.browse()["total"], 0)
        self.assertIsNone(self.catalog.find("before-1"))

    def test_invalid_json_is_reported_and_repaired_file_can_reload(self):
        self.write()
        self.assertEqual(self.catalog.browse()["total"], 1)
        self.path.write_text("{not json}", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            self.catalog.browse()
        self.write("repaired")
        self.assertEqual(self.catalog.browse()["samples"][0]["id"], "repaired")


if __name__ == "__main__":
    unittest.main(verbosity=2)
