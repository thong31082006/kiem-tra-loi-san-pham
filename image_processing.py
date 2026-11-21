import cv2
import numpy as np


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray, k=5):
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(gray, (k, k), 0)


def canny_edges(gray):
    return cv2.Canny(gaussian_blur(gray, 5), 60, 150)


def _largest_valid_contour(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary.shape
    area_img = h * w
    valid = []
    for c in contours:
        a = cv2.contourArea(c)
        if area_img * 0.03 <= a <= area_img * 0.95:
            x, y, bw, bh = cv2.boundingRect(c)
            cx, cy = x + bw / 2, y + bh / 2
            dist = ((cx - w / 2) / w) ** 2 + ((cy - h / 2) / h) ** 2
            valid.append((a * (1.0 - min(dist, 0.75)), c))
    return max(valid, key=lambda t: t[0])[1] if valid else None


def segment_product(image):
    """Tách vật thể chính khỏi nền. Tốt nhất với nền đơn sắc tương phản."""
    gray = gaussian_blur(to_gray(image), 5)
    _, n = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cn, ci = _largest_valid_contour(n), _largest_valid_contour(inv)
    if cn is None and ci is None:
        raise ValueError("Không tách được sản phẩm. Hãy chụp trên nền đơn giản, đủ sáng và để sản phẩm nằm trọn khung hình.")
    def score(c, shape):
        if c is None:
            return -1
        h, w = shape
        a = cv2.contourArea(c)
        x, y, bw, bh = cv2.boundingRect(c)
        center = abs((x + bw/2)-w/2)/w + abs((y+bh/2)-h/2)/h
        return a * (1.2 - min(center, 1.0))
    c = cn if score(cn, n.shape) >= score(ci, inv.shape) else ci
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    return mask, c, cv2.boundingRect(c)


def normalized_mask(mask, bbox, size=(420, 520)):
    x, y, w, h = bbox
    crop = mask[y:y+h, x:x+w]
    if crop.size == 0:
        raise ValueError("Vùng sản phẩm không hợp lệ.")
    return cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST)


def normalized_crop(image, bbox, size=(420, 520)):
    x, y, w, h = bbox
    crop = image[y:y+h, x:x+w]
    if crop.size == 0:
        raise ValueError("Vùng sản phẩm không hợp lệ.")
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def mask_iou(ref_mask, ref_bbox, test_mask, test_bbox):
    r = normalized_mask(ref_mask, ref_bbox)
    t = normalized_mask(test_mask, test_bbox)
    inter = np.logical_and(r > 0, t > 0).sum()
    union = np.logical_or(r > 0, t > 0).sum()
    return float(inter / max(union, 1) * 100.0), r, t, cv2.absdiff(r, t)


def make_visual_mask(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def make_visual_gray(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
