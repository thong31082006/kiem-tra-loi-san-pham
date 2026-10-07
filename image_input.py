from pathlib import Path
import cv2
import numpy as np


def load_image(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")
    data = np.fromfile(str(p), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không thể đọc ảnh. Hãy chọn file JPG/PNG/BMP hợp lệ.")
    return image


def resize_for_processing(image, max_side=1100):
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return image.copy()
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def capture_from_camera(camera_index=0, window_title="Camera - Kiem tra san pham"):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    captured = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        preview = frame.copy()
        cv2.putText(preview, "SPACE: chup | ESC: huy", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window_title, preview)
        key = cv2.waitKey(20) & 0xFF
        if key == 32:
            captured = frame.copy()
            break
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return captured
