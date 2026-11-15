"""Locate one bottle before custom quality classification; no reference image."""
import os
import threading

import cv2

from .features import APP_DIR, as_rgb_pil, bottle_crop_bounds, crop_bottle_bbox


class BottleLocator:
    def __init__(self):
        self.network = None
        self.bottle_class = None
        self.lock = threading.RLock()

    def load(self):
        if self.network is None:
            import torch
            from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large

            torch.set_num_threads(min(4, os.cpu_count() or 1))
            torch.hub.set_dir(str(APP_DIR / "models" / "pretrained"))
            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.network = ssdlite320_mobilenet_v3_large(weights=weights).eval()
            self.bottle_class = weights.meta["categories"].index("bottle")
        return self.network

    def locate(self, image):
        import torch
        from torchvision.transforms.functional import to_tensor

        with self.lock:
            network = self.load()
            h, w = image.shape[:2]
            scale = min(1.0, 960 / max(h, w))
            small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale)))) if scale < 1 else image
            with torch.inference_mode():
                result = network([to_tensor(as_rgb_pil(small))])[0]
            candidates = []
            for box, label, score in zip(result["boxes"].tolist(), result["labels"].tolist(), result["scores"].tolist()):
                x1, y1, x2, y2 = [value / scale for value in box]
                if label == self.bottle_class and score >= 0.30 and (x2 - x1) * (y2 - y1) >= h * w * 0.015:
                    candidates.append((score, [x1, y1, x2, y2]))
            if not candidates:
                raise ValueError("Không tìm thấy chai đủ rõ trong ảnh. Hãy chụp trọn một chai, gồm nắp và đáy.")
            candidates = self.suppress_nested(candidates)
            if len(candidates) > 1:
                raise ValueError("Ảnh có nhiều chai. Hãy chọn hoặc chụp riêng một chai để kiểm tra.")
            score, box = candidates[0]
            x1, y1, x2, y2 = box
            normalized = [(x1 + x2) / (2 * w), (y1 + y2) / (2 * h), (x2 - x1) / w, (y2 - y1) / h]
            return crop_bottle_bbox(image, normalized), {
                "bbox": [round(value, 1) for value in box], "score": score,
                "crop_bbox": bottle_crop_bounds(image, normalized),
            }

    @staticmethod
    def suppress_nested(candidates):
        """SSD can label the neck and whole body separately; retain one containing box."""
        kept = []
        for candidate in sorted(candidates, reverse=True):
            _, box = candidate
            duplicate = False
            for _, previous in kept:
                intersection = max(0, min(box[2], previous[2]) - max(box[0], previous[0])) * max(0, min(box[3], previous[3]) - max(box[1], previous[1]))
                area = (box[2] - box[0]) * (box[3] - box[1])
                other_area = (previous[2] - previous[0]) * (previous[3] - previous[1])
                if intersection / max(1, min(area, other_area)) >= 0.8:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept
