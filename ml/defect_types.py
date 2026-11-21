"""Labels, validation policies and metrics for the single six-class SVM."""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

TYPE_CODES = ("normal", "defect_low_fill", "defect_no_cap", "defect_loose_cap", "defect_debris", "defect_label_damage")
TYPE_NAMES = {
    "normal": "Không thấy lỗi trong các nhóm đã học",
    "defect_low_fill": "Thiếu mức nước",
    "defect_no_cap": "Thiếu nắp",
    "defect_loose_cap": "Nắp lỏng / hỏng",
    "defect_debris": "Dị vật nhìn thấy",
    "defect_label_damage": "Nhãn hỏng / lệch / thiếu",
}
UNSUPPORTED_TYPES = ["Móp / biến dạng chai", "Nước đục / đổi màu"]
TYPE_SCOPE = (
    "Nhận diện loại lỗi chính trên mẫu chai Ron 88. Nhãn hỏng, lệch và thiếu được nguồn gộp chung. "
    "Chưa kiểm chứng nhiều lỗi đồng thời, hãng chai khác hoặc lỗi bị khuất. "
    "Chưa hỗ trợ móp chai và nước đục; ảnh không xác nhận được độ kín thực tế của nắp."
)


def classify_type_scores(scores, policies):
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(TYPE_CODES),) or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1, atol=1e-4):
        raise ValueError("Điểm phân loại lỗi không hợp lệ.")
    index = int(values.argmax())
    code, score = TYPE_CODES[index], float(values[index])
    policy = policies.get(code, {})
    accepted = score >= float(policy.get("threshold", 1.01)) and policy.get("enabled", False)
    return {
        "status": ("normal" if index == 0 else "typed") if accepted else "uncertain",
        "code": code if accepted else None, "name": TYPE_NAMES[code] if accepted else "Chưa xác định được loại lỗi",
        "score": score, "threshold": policy.get("threshold"),
        "scores": {key: float(value) for key, value in zip(TYPE_CODES, values)},
        "scope": TYPE_SCOPE, "unsupported": UNSUPPORTED_TYPES,
    }


def raw_metrics(y, scores):
    predicted = np.asarray(scores).argmax(axis=1)
    return {"accuracy": float((predicted == y).mean()),
            "macro_f1": float(f1_score(y, predicted, average="macro", labels=list(range(6)), zero_division=0)),
            "confusion_matrix": confusion_matrix(y, predicted, labels=list(range(6))).tolist(),
            "per_class": classification_report(y, predicted, labels=list(range(6)), target_names=list(TYPE_CODES), output_dict=True, zero_division=0)}


def calibrate(y, scores):
    predicted = scores.argmax(axis=1)
    policies = {"normal": {"enabled": True, "threshold": 0.65}}
    for index, code in enumerate(TYPE_CODES[1:], 1):
        best = None
        for threshold in np.linspace(0.5, 0.95, 19):
            positive = (predicted == index) & (scores[:, index] >= threshold)
            tp = int(((y == index) & positive).sum())
            fp = int(((y != index) & positive).sum())
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, int((y == index).sum()))
            if precision < 0.85 or recall < 0.2 or tp < 3:
                continue
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            candidate = {"enabled": True, "threshold": float(threshold), "validation_precision": precision,
                         "validation_recall": recall, "validation_f1": f1, "validation_tp": tp, "validation_fp": fp}
            rank = (f1, recall, precision, -float(threshold))
            if best is None or rank > best[0]:
                best = (rank, candidate)
        policies[code] = best[1] if best else {"enabled": False, "threshold": 1.01, "reason": "Chưa đạt Precision 85% và Recall 20% trên Validation."}
    return policies


def accepted_metrics(y, scores, policies):
    outcomes = [classify_type_scores(row, policies) for row in scores]
    predicted = np.array([TYPE_CODES.index(row["code"]) if row["code"] else 6 for row in outcomes])
    per_class = {}
    for index, code in enumerate(TYPE_CODES):
        tp = int(((y == index) & (predicted == index)).sum())
        fp = int(((y != index) & (predicted == index)).sum())
        total = int((y == index).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, total)
        per_class[code] = {"precision": precision, "recall": recall,
                           "f1": 2 * precision * recall / max(1e-12, precision + recall),
                           "support": total, "tp": tp, "fp": fp, "fn_or_uncertain": total - tp}
    return {"accuracy": float((predicted == y).mean()), "coverage": float((predicted != 6).mean()),
            "uncertain_count": int((predicted == 6).sum()), "per_class": per_class,
            "confusion_matrix": confusion_matrix(y, predicted, labels=list(range(7))).tolist()[:6],
            "column_labels": list(TYPE_CODES) + ["uncertain"]}
