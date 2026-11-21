"""Checks for the single SVM pipeline, quality rejection and real trained artifacts."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from ml.features import read_bgr, read_model_image
from ml.hog_svm import ARTIFACT_ROOT, HogSvmService, extract_hog, quality_result
from ml.train import DEFAULT_MANIFEST
from web_history import HistoryStore


class HogSvmTests(unittest.TestCase):
    def test_hog_is_grayscale_only_and_has_3780_values(self):
        # Different BGR colors with identical OpenCV gray luminance must give
        # identical HOG, demonstrating this path has no appended HSV histogram.
        image = np.zeros((128, 64, 3), dtype=np.uint8)
        image[24:90, 15:40] = (0, 0, 255)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vector = extract_hog(image)
        np.testing.assert_array_equal(vector, extract_hog(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))
        self.assertEqual(vector.shape, (3780,))
        self.assertEqual(vector.dtype, np.float32)

    def test_quality_sums_defect_classes_and_retains_uncertain(self):
        result = quality_result([0.1, 0.18, 0.18, 0.18, 0.18, 0.18], 0.5)
        self.assertEqual(result['label'], 'defect')
        self.assertAlmostEqual(result['confidence'], 0.9)
        self.assertEqual(quality_result([0.55, 0.45, 0, 0, 0, 0], 0.5)['label'], 'uncertain')
        self.assertEqual(quality_result([0.9, 0.1, 0, 0, 0, 0], 0.5)['label'], 'normal')

    def test_missing_and_escaping_artifacts_never_become_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            service = HogSvmService(temp)
            self.assertFalse(service.status()['ready'])
            with self.assertRaises(FileNotFoundError):
                service.predict(np.zeros((64, 64, 3), dtype=np.uint8))
            Path(temp, 'latest.json').write_text(json.dumps({'run_id': '../outside'}))
            self.assertFalse(service.status()['ready'])
            self.assertIn('error', service.status())

    @unittest.skipUnless((ARTIFACT_ROOT / 'latest.json').is_file() and DEFAULT_MANIFEST.is_file(),
                         'Requires trained local HOG + SVM and the source dataset')
    def test_real_model_api_default_and_identical_pixels_under_different_names(self):
        # Temporary history avoids adding any test records to the user's DB.
        with tempfile.TemporaryDirectory() as temp:
            store = HistoryStore(Path(temp) / 'history')
            with patch('web_history.HistoryStore', return_value=store):
                api = importlib.import_module('web_api')
            service = HogSvmService()
            manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding='utf-8'))
            records = [r for r in manifest['records'] if r['split'] == 'test']
            record = next(r for r in records if r['label'] == 'normal')
            image = read_bgr(record['path'])
            ok, encoded = cv2.imencode('.png', image)
            self.assertTrue(ok)
            with TestClient(api.create_app(service, store)) as client:
                self.assertEqual([m['id'] for m in client.get('/api/models').json()['models']], ['svm'])
                first = client.post('/api/predict', files={'file': ('normal.png', encoded.tobytes(), 'image/png')})
                second = client.post('/api/predict', files={'file': ('defect_no_cap.png', encoded.tobytes(), 'image/png')})
                self.assertEqual(first.status_code, 200, first.text)
                self.assertEqual(second.status_code, 200, second.text)
                self.assertEqual(first.json()['probabilities'], second.json()['probabilities'])
                self.assertEqual(first.json()['model'], 'svm')
                self.assertEqual(client.post('/api/predict', data={'model': 'mobilenet'},
                                             files={'file': ('a.png', encoded.tobytes(), 'image/png')}).status_code, 422)
            # Same training crop and same feature function as the service uses.
            crop = read_model_image(record)
            expected = service._loaded.predict_proba(extract_hog(crop).reshape(1, -1))[0]
            self.assertAlmostEqual(first.json()['probabilities']['normal'], float(expected[0]), places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
