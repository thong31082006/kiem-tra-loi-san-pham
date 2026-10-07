import cv2
import numpy as np
from models import InspectionResult
from image_processing import segment_product, mask_iou, normalized_crop, normalized_mask, canny_edges, make_visual_mask

PRODUCT_CODE = "TILE"
PRODUCT_NAME = "Gạch men"


def _crack_ratio(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(25,25)))
    inside = cv2.erode(mask, np.ones((19,19),np.uint8), iterations=1)
    blackhat = cv2.bitwise_and(blackhat, inside)
    crack = cv2.threshold(blackhat, 35, 255, cv2.THRESH_BINARY)[1]
    crack = cv2.morphologyEx(crack, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
    ratio = float((crack>0).sum()/max((inside>0).sum(),1)*100)
    return ratio, crack


def inspect(reference, test):
    ref_mask, _, ref_bbox = segment_product(reference)
    test_mask, test_contour, test_bbox = segment_product(test)
    shape, rn, tn, diff = mask_iou(ref_mask, ref_bbox, test_mask, test_bbox)

    ref_cr, _ = _crack_ratio(reference, ref_mask)
    test_cr, crack_mask = _crack_ratio(test, test_mask)
    crack_delta = max(0.0, test_cr-ref_cr)

    # Bất thường bề mặt theo sai khác màu sau chuẩn hóa.
    rc = normalized_crop(reference, ref_bbox)
    tc = normalized_crop(test, test_bbox)
    rl = cv2.cvtColor(rc, cv2.COLOR_BGR2LAB)
    tl = cv2.cvtColor(tc, cv2.COLOR_BGR2LAB)
    d = cv2.absdiff(cv2.GaussianBlur(rl,(5,5),0), cv2.GaussianBlur(tl,(5,5),0))
    mag = np.linalg.norm(d.astype(np.float32), axis=2)
    inter = np.logical_and(rn>0, tn>0)
    stain_ratio = float(np.logical_and(mag>38, inter).sum()/max(inter.sum(),1)*100)

    defects, scores = [], {}
    if shape < 96.0:
        defects.append("Mẻ cạnh / sai hình dạng viên gạch")
        scores[defects[-1]] = 100-shape
    if crack_delta > 0.45:
        defects.append("Nứt bề mặt")
        scores[defects[-1]] = crack_delta
    if stain_ratio > 4.0:
        defects.append("Vết bẩn / bất thường màu bề mặt")
        scores[defects[-1]] = stain_ratio

    out = test.copy()
    cv2.drawContours(out, [test_contour], -1, (255,120,0), 2)
    x,y,w,h = test_bbox
    if shape < 96.0:
        cv2.rectangle(out,(x,y),(x+w,y+h),(0,0,255),3)
    if crack_delta > 0.45:
        # Crack mask tọa độ gốc nên có thể box trực tiếp.
        cnts,_ = cv2.findContours(crack_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts,key=cv2.contourArea,reverse=True)[:8]:
            if cv2.contourArea(c) < 8: continue
            bx,by,bw,bh=cv2.boundingRect(c)
            if bw*bh>60:
                cv2.rectangle(out,(bx,by),(bx+bw,by+bh),(255,0,255),2)
    if stain_ratio > 4.0:
        cv2.putText(out,"SURFACE",(x,max(20,y+h-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,80,255),2)

    edge = canny_edges(cv2.cvtColor(test, cv2.COLOR_BGR2GRAY))
    edge = cv2.bitwise_and(edge,test_mask)
    status = "KHÔNG ĐẠT" if defects else "ĐẠT"
    metrics = {
        "Độ giống hình dáng": round(shape,2),
        "Nứt tăng so chuẩn (%)": round(crack_delta,2),
        "Bất thường màu bề mặt (%)": round(stain_ratio,2),
    }
    return InspectionResult(status, defects, metrics, out, make_visual_mask(test_mask), cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR), cv2.cvtColor(diff,cv2.COLOR_GRAY2BGR), scores)
