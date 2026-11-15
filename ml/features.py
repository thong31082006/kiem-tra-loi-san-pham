"""Pixel-only features and decoding shared by training and web inference."""
from pathlib import Path

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("normal", "defect")
FEATURE_VERSION = "auto-bottle-roi-hog64x128-hsv48-v3"
IMAGE_SIZE = 224
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def image_path(value):
    path = Path(value)
    return path if path.is_absolute() else APP_DIR / path


def read_bgr(value):
    path = image_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {path.name}")
    return image


def validate_image(image):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("Cần một ảnh màu BGR hợp lệ.")
    if image.dtype != np.uint8:
        raise ValueError("Ảnh cần có kiểu uint8.")
    return image


def read_model_image(record):
    """Use the identical automatic crop for fitting and inference preprocessing."""
    if record.get("training_path"):
        return read_bgr(record["training_path"])
    image = read_bgr(record["path"])
    return crop_bottle_bbox(image, record["bbox"]) if record.get("bbox") else image


def extract_features(image):
    """HOG describes shape/edges; HSV histograms add overall color information."""
    validate_image(image)
    resized = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    edges = hog.compute(gray).reshape(-1)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histograms = []
    for channel, limit in ((0, 180), (1, 256), (2, 256)):
        histogram = cv2.calcHist([hsv], [channel], None, [16], [0, limit]).reshape(-1)
        histogram /= max(float(histogram.sum()), 1.0)
        histograms.append(histogram)
    return np.concatenate([edges, *histograms]).astype(np.float32)


def bottle_crop_bounds(image, bbox, padding=0.08):
    """Pixel bounds shared by inference and its visual explanation."""
    validate_image(image)
    if len(bbox) != 4 or not all(np.isfinite(value) for value in bbox):
        raise ValueError("Vùng chai không hợp lệ.")
    cx, cy, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        raise ValueError("Vùng chai không hợp lệ.")
    height, width = image.shape[:2]
    x1 = max(0, int((cx - bw * (0.5 + padding)) * width))
    x2 = min(width, int((cx + bw * (0.5 + padding)) * width))
    y1 = max(0, int((cy - bh * (0.5 + padding)) * height))
    y2 = min(height, int((cy + bh * (0.5 + padding)) * height))
    if min(y2 - y1, x2 - x1) < 10:
        raise ValueError("Vùng chai quá nhỏ để kiểm tra.")
    return [x1, y1, x2, y2]


def crop_bottle_bbox(image, bbox, padding=0.08):
    """Crop a normalized box, retaining some context for the cap and base."""
    x1, y1, x2, y2 = bottle_crop_bounds(image, bbox, padding)
    crop = image[y1:y2, x1:x2]
    return crop.copy()


def cnn_transform(augment=False):
    # Torchvision stays lazy so classical models can run without loading PyTorch.
    from torchvision import transforms

    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if augment:
        steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12, fill=(127, 127, 127)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        ])
    steps.extend([transforms.ToTensor(), transforms.Normalize(IMAGE_MEAN, IMAGE_STD)])
    return transforms.Compose(steps)


def as_rgb_pil(image):
    from PIL import Image

    validate_image(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
