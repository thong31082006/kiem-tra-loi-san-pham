import cv2
import numpy as np
from models import InspectionResult
from image_processing import canny_edges, contour_boxes, make_visual_mask

PRODUCT_CODE = "HOE"
PRODUCT_NAME = "Cuốc sắt"
HOE_MIN_SHAPE_SIMILARITY = 90.0
HOE_MAX_RUST_RATIO = 1.0
HOE_MAX_SCRATCH_DELTA = 2.0


def _normalize_hoe(image, mask, angle=0.0, size=(512, 512)):
    """Rotate on a padded canvas, crop and center with uniform scaling."""
    side = int(np.ceil(np.hypot(*mask.shape))) + 4
    canvas = np.zeros((side, side), np.uint8)
    photo = np.full((side, side, 3), 255, np.uint8)
    h, w = mask.shape
    x, y = (side-w)//2, (side-h)//2
    canvas[y:y+h, x:x+w] = mask
    photo[y:y+h, x:x+w] = image
    matrix = cv2.getRotationMatrix2D((side/2, side/2), angle, 1)
    canvas = cv2.warpAffine(canvas, matrix, (side, side), flags=cv2.INTER_NEAREST)
    photo = cv2.warpAffine(photo, matrix, (side, side), borderValue=(255,255,255))
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    scale = min((size[0]-32)/w, (size[1]-32)/h)
    target = (max(1,round(w*scale)), max(1,round(h*scale)))
    crop = cv2.resize(photo[y:y+h,x:x+w], target, interpolation=cv2.INTER_AREA)
    crop_mask = cv2.resize(canvas[y:y+h,x:x+w], target, interpolation=cv2.INTER_NEAREST)
    result = np.full((size[1],size[0],3),255,np.uint8)
    result_mask = np.zeros((size[1],size[0]),np.uint8)
    x,y = (size[0]-target[0])//2, (size[1]-target[1])//2
    result[y:y+target[1],x:x+target[0]] = crop
    result_mask[y:y+target[1],x:x+target[0]] = crop_mask
    result[result_mask == 0] = 255
    return result, result_mask


def _segment_hoe(image):
    """Select the largest foreground contour without a center-position bias."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(5,5),0)
    binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    candidates = []
    height,width = gray.shape
    for binary in (binary,cv2.bitwise_not(binary)):
        contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if x > 0 and y > 0 and x+w < width and y+h < height and cv2.contourArea(c) > 25:
                candidates.append(c)
    if not candidates:
        raise ValueError("Không tách được cuốc. Dùng nền tương phản và để vật thể nằm trọn khung hình.")
    c = max(candidates,key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask,[c],-1,255,cv2.FILLED)
    return mask,c


def _contour_angle(contour):
    _, axes = cv2.PCACompute(contour.reshape(-1,2).astype(np.float64), mean=None)
    return float(np.degrees(np.arctan2(axes[0,1], axes[0,0])))


def _align_hoe(reference, test):
    rm,rc = _segment_hoe(reference)
    tm,tc = _segment_hoe(test)
    reference,rm = _normalize_hoe(reference,rm)
    angle = (_contour_angle(tc)-_contour_angle(rc)+90)%180-90
    angle = angle if abs(angle) <= 20 else 0.0
    best = None
    # Contour sampling and a chipped edge can bias PCA. Refine across the
    # mild-rotation range as well as around the estimated contour direction.
    corrections = np.unique(np.concatenate((np.arange(-20, 21, 1), np.arange(angle-2, angle+2.1, 0.5))))
    for correction in corrections:
        image,mask = _normalize_hoe(test,tm,correction)
        inter = np.logical_and(rm>0,mask>0).sum()
        union = np.logical_or(rm>0,mask>0).sum()
        shape = float(inter/max(union,1)*100)
        if best is None or shape > best[0]:
            best = shape,image,mask
    shape,test,tm = best
    return reference,test,rm,tm,shape,cv2.absdiff(rm,tm)


def _rust_mask(image, product_mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rust = cv2.inRange(hsv, np.array([3, 55, 35], np.uint8), np.array([28, 255, 235], np.uint8))
    rust = cv2.bitwise_and(rust, product_mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(cv2.morphologyEx(rust, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k, iterations=2)


def _scratch_mask(image, product_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21)))
    inside = cv2.erode(product_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17,17)), iterations=1)
    blackhat = cv2.bitwise_and(blackhat, inside)
    m = cv2.threshold(blackhat, 45, 255, cv2.THRESH_BINARY)[1]
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8)), inside


def inspect(reference, test):
    reference,test,ref_mask,test_mask,shape,diff_norm = _align_hoe(reference,test)
    contours,_ = cv2.findContours(test_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    test_contour = max(contours,key=cv2.contourArea)
    test_bbox = cv2.boundingRect(test_contour)

    rust = _rust_mask(test, test_mask)
    rust_ratio = float((rust > 0).sum() / max((test_mask > 0).sum(), 1) * 100)

    rs, ri = _scratch_mask(reference, ref_mask)
    ts, ti = _scratch_mask(test, test_mask)
    rr = (rs > 0).sum() / max((ri > 0).sum(), 1) * 100
    tr = (ts > 0).sum() / max((ti > 0).sum(), 1) * 100
    scratch_delta = float(max(0, tr - rr))

    defects, scores = [], {}
    if shape < HOE_MIN_SHAPE_SIMILARITY:
        defects.append("Mẻ / biến dạng lưỡi cuốc")
        scores[defects[-1]] = 100 - shape
    if rust_ratio > HOE_MAX_RUST_RATIO:
        defects.append("Gỉ sét bề mặt")
        scores[defects[-1]] = rust_ratio
    if scratch_delta > HOE_MAX_SCRATCH_DELTA:
        defects.append("Nứt / trầy bất thường")
        scores[defects[-1]] = scratch_delta

    out = test.copy()
    cv2.drawContours(out, [test_contour], -1, (255, 120, 0), 2)
    if rust_ratio > HOE_MAX_RUST_RATIO:
        for x,y,w,h in contour_boxes(rust, 60, 6):
            cv2.rectangle(out, (x,y), (x+w,y+h), (0,80,255), 3)
            cv2.putText(out, "RUST", (x,max(20,y-6)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,80,255), 2)
    if scratch_delta > HOE_MAX_SCRATCH_DELTA:
        for x,y,w,h in contour_boxes(ts, 18, 8):
            if w*h >= 70:
                cv2.rectangle(out, (x,y), (x+w,y+h), (255,0,255), 2)
    # Shape defect: mark bbox if large difference rather than mapping every pixel.
    if shape < HOE_MIN_SHAPE_SIMILARITY:
        x,y,w,h = test_bbox
        cv2.rectangle(out, (x,y), (x+w,y+h), (0,0,255), 3)

    edge = canny_edges(cv2.cvtColor(test, cv2.COLOR_BGR2GRAY))
    edge = cv2.bitwise_and(edge, test_mask)
    diff_vis = cv2.cvtColor(diff_norm, cv2.COLOR_GRAY2BGR)
    status = "ĐẠT" if (shape >= HOE_MIN_SHAPE_SIMILARITY and rust_ratio <= HOE_MAX_RUST_RATIO and scratch_delta <= HOE_MAX_SCRATCH_DELTA) else "KHÔNG ĐẠT"
    metrics = {
        "Độ giống hình dáng": round(shape, 2),
        "Tỷ lệ gỉ sét (%)": round(rust_ratio, 2),
        "Nứt/trầy tăng so chuẩn (%)": round(scratch_delta, 2),
    }
    return InspectionResult(status, defects, metrics, out, make_visual_mask(test_mask), cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR), diff_vis, scores)
