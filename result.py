from pathlib import Path
from datetime import datetime
import cv2
from database import save_inspection


def imwrite_unicode(path: Path, image):
    suffix = path.suffix.lower() or ".jpg"
    ok, data = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError("Không thể mã hóa ảnh kết quả.")
    data.tofile(str(path))


def save_result(product_name, reference_path, test_path, inspection_result):
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_product = product_name.lower().replace(" ", "_").replace("ố", "o").replace("ạ", "a")
    out_path = out_dir / f"{safe_product}_{inspection_result.status.replace(' ', '_')}_{stamp}.jpg"
    imwrite_unicode(out_path, inspection_result.result_image)
    save_inspection(
        product_name=product_name,
        reference_path=reference_path,
        image_name=Path(test_path).name if test_path and test_path != "camera" else "camera.jpg",
        image_path=test_path or "camera",
        status=inspection_result.status,
        metrics=inspection_result.metrics,
        defects=inspection_result.defects,
        defect_scores=inspection_result.defect_scores,
        result_path=str(out_path),
    )
    return str(out_path)
