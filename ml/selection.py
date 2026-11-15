"""Validation-only selection accounting for the same abstention policy as the app."""
import numpy as np


def pipeline_metrics(labels, probabilities, threshold, uncertainty=0.65):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    raw = (probabilities >= threshold).astype(int)
    confidence = np.where(raw == 1, probabilities, 1 - probabilities)
    predicted = np.where(confidence >= uncertainty, raw, 2)
    matrix = np.zeros((2, 3), dtype=int)
    np.add.at(matrix, (labels, predicted), 1)
    normal_total, defect_total = matrix.sum(axis=1)
    tn, fp, normal_uncertain = matrix[0]
    fn, tp, defect_uncertain = matrix[1]
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / defect_total) if defect_total else 0.0
    specificity = float(tn / normal_total) if normal_total else 0.0
    return {
        "confusion_matrix": matrix.tolist(),
        "accuracy": float((tn + tp) / len(labels)),
        "balanced_correct": (specificity + recall) / 2,
        "precision": precision, "recall": recall, "normal_recall": specificity,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positives": int(fp), "false_negatives": int(fn),
        "uncertain_count": int(normal_uncertain + defect_uncertain),
        "coverage": float((predicted != 2).mean()),
    }


def selection_rank(summary):
    # Both classes count equally; abstentions count as failures, not successes.
    return (summary["balanced_correct"], summary["precision"], summary["recall"], summary["coverage"])


def choose_pipeline_threshold(labels, probabilities, uncertainty=0.65):
    best = None
    for threshold in np.linspace(0.1, 0.9, 33):
        summary = pipeline_metrics(labels, probabilities, float(threshold), uncertainty)
        rank = (*selection_rank(summary), -abs(float(threshold) - 0.5))
        if best is None or rank > best[0]:
            best = (rank, float(threshold), summary)
    return best[1], best[2]


def accept_candidate(candidate, baseline):
    """Do not reduce false alarms by simply refusing more images or missing defects."""
    return (
        candidate["balanced_correct"] > baseline["balanced_correct"] + 1e-12
        and candidate["false_positives"] <= baseline["false_positives"]
        and candidate["recall"] >= baseline["recall"] - 0.02
    )
