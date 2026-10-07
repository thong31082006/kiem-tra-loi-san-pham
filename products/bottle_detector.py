import cv2
import numpy as np
from models import InspectionResult
from image_processing import segment_product, mask_iou, normalized_mask, normalized_crop, canny_edges, make_visual_mask

PRODUCT_CODE = "BOTTLE"
PRODUCT_NAME = "Chai nhựa"


def inspect(reference, test):
    ref_mask, _, ref_bbox = segment_product(reference)
    test_mask, test_contour, test_bbox = segment_product(test)
    shape, ref_n, test_n, diff = mask_iou(ref_mask, ref_bbox, test_mask, test_bbox)

    # Vùng 15% phía trên dùng như dấu hiệu nắp/cổ chai.
    top_h = max(20, int(ref_n.shape[0] * 0.15))
    ref_top = (ref_n[:top_h] > 0).sum()
    test_top = (test_n[:top_h] > 0).sum()
    cap_ratio = float(test_top / max(ref_top, 1) * 100.0)

    # So sánh bề mặt sau khi chuẩn hóa; bỏ vùng nền bằng mask giao nhau.
    rc = normalized_crop(reference, ref_bbox)
    tc = normalized_crop(test, test_bbox)
    rg = cv2.cvtColor(rc, cv2.COLOR_BGR2GRAY)
    tg = cv2.cvtColor(tc, cv2.COLOR_BGR2GRAY)
    inter = cv2.bitwise_and(ref_n, test_n)
    ad = cv2.absdiff(cv2.GaussianBlur(rg,(5,5),0), cv2.GaussianBlur(tg,(5,5),0))
    ad = cv2.bitwise_and(ad, inter)
    surface = cv2.threshold(ad, 42, 255, cv2.THRESH_BINARY)[1]
    surface = cv2.morphologyEx(surface, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    surface_ratio = float((surface > 0).sum() / max((inter > 0).sum(), 1) * 100.0)

    defects, scores = [], {}
    if cap_ratio < 95.0:
        defects.append("Thiếu / sai nắp chai")
        scores[defects[-1]] = 100 - cap_ratio
    if shape < 93.0:
        defects.append("Thân chai méo / sai hình dạng")
        scores[defects[-1]] = 100 - shape
    if surface_ratio > 8.0:
        defects.append("Bề mặt chai bất thường")
        scores[defects[-1]] = surface_ratio

    out = test.copy()
    cv2.drawContours(out, [test_contour], -1, (255,120,0), 2)
    x,y,w,h = test_bbox
    if cap_ratio < 95.0:
        cv2.rectangle(out, (x, y), (x+w, y+max(10,int(h*0.18))), (0,0,255), 3)
        cv2.putText(out, "CAP", (x,max(20,y-6)), cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
    if shape < 93.0:
        cv2.rectangle(out, (x,y), (x+w,y+h), (0,0,255), 2)
    if surface_ratio > 8.0:
        cv2.putText(out, "SURFACE", (x,max(20,y+h-8)), cv2.FONT_HERSHEY_SIMPLEX,.55,(255,0,255),2)

    edge = canny_edges(cv2.cvtColor(test, cv2.COLOR_BGR2GRAY))
    edge = cv2.bitwise_and(edge, test_mask)
    status = "KHÔNG ĐẠT" if defects else "ĐẠT"
    metrics = {
        "Độ giống hình dáng": round(shape,2),
        "Tỷ lệ vùng nắp so chuẩn (%)": round(cap_ratio,2),
        "Bất thường bề mặt (%)": round(surface_ratio,2),
    }
    return InspectionResult(status, defects, metrics, out, make_visual_mask(test_mask), cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR), cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR), scores)
