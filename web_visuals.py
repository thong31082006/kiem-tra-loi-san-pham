"""Visualize the actual automatic crop and OpenCV HOG, never defect attribution."""
import base64

import cv2
import numpy as np

from ml.features import extract_features


def png_url(image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Không tạo được ảnh minh họa.")
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def hog_cells(crop):
    # OpenCV orders both blocks and cells by x then y (hog.cpp, HOGCache).
    blocks = extract_features(crop)[:3780].reshape(7, 15, 2, 2, 9)
    cells = np.zeros((8, 16, 9), np.float32)
    counts = np.zeros((8, 16, 1), np.float32)
    for x in range(7):
        for y in range(15):
            cells[x:x + 2, y:y + 2] += blocks[x, y]
            counts[x:x + 2, y:y + 2] += 1
    return cells / counts


def hog_image(crop):
    cells = hog_cells(crop)
    peak = max(float(cells.max()), 1e-6)
    canvas = np.full((512, 256, 3), (35, 27, 19), np.uint8)
    for x in range(8):
        for y in range(16):
            center = np.array([x * 32 + 16, y * 32 + 16])
            for index, strength in enumerate(cells[x, y]):
                if strength <= 0:
                    continue
                angle = (index + 0.5) * np.pi / 9
                delta = np.array([np.cos(angle), np.sin(angle)]) * (14 * strength / peak)
                a, b = np.rint(center - delta).astype(int), np.rint(center + delta).astype(int)
                shade = int(65 + 190 * strength / peak)
                cv2.line(canvas, tuple(a), tuple(b), (shade, shade, shade), 1, cv2.LINE_AA)
    return canvas


def build_visuals(image, localization):
    bounds = (localization or {}).get("crop_bbox")
    if not bounds:
        return None
    x1, y1, x2, y2 = map(int, bounds)
    height, width = image.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    crop = image[y1:y2, x1:x2]
    return {
        "crop_bbox": bounds, "image_size": [width, height],
        "crop": png_url(crop), "hog": png_url(hog_image(crop)),
        "hog_features": 3780, "hsv_features": 48,
    }
