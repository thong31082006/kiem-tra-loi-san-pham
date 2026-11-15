"""Evaluate real full-image inference, including localization and abstention."""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.features import APP_DIR, read_bgr
from ml.inference import DEFAULT_ARTIFACT_ROOT, MODEL_NAMES, ModelService


LOCALIZATION_REJECTIONS = (
    "Không tìm thấy chai đủ rõ",
    "Ảnh có nhiều chai",
    "Vùng chai quá nhỏ",
)


def summarize(rows):
    matrix = [[0, 0, 0], [0, 0, 0]]
    for row in rows:
        matrix[("normal", "defect").index(row["truth"])][("normal", "defect", "uncertain").index(row["label"])] += 1
    total = len(rows)
    true_positive, false_positive = matrix[1][1], matrix[0][1]
    defect_total = sum(matrix[1])
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / defect_total if defect_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    correct = matrix[0][0] + matrix[1][1]
    uncertain = matrix[0][2] + matrix[1][2]
    return {
        "total": total, "correct_count": correct,
        "accuracy": correct / total if total else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "coverage": (total - uncertain) / total if total else 0.0,
        "uncertain_count": uncertain,
        "localization_rejected_count": sum(row["localization_rejected"] for row in rows),
        "confusion_matrix": matrix,
        "row_labels": ["normal", "defect"],
        "column_labels": ["normal", "defect", "uncertain"],
    }


def evaluate(manifest_path, artifact_root, max_samples=None, output=None):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    records = [row for row in manifest.get("records", []) if row["split"] == "test"]
    if not records:
        raise ValueError("Manifest không có ảnh Test.")
    if any(row.get("label") not in ("normal", "defect") for row in records):
        raise ValueError("Tập Test có nhãn không hợp lệ.")
    available = len(records)
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max-samples phải lớn hơn 0.")
        records = records[:max_samples]
    artifact_root = Path(artifact_root).resolve()
    service = ModelService(artifact_root, enable_defect_types=False)
    status = service.status()
    if not status.get("ready"):
        raise RuntimeError(f"Cần cả ba mô hình đã sẵn sàng: {status.get('error', 'missing artifacts')}")
    run_id = status["run_id"]
    run_dir = (artifact_root / run_id).resolve()
    run_dir.relative_to(artifact_root)
    training_report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if training_report.get("input_mode") not in ("bottle_roi", "auto_bottle_roi"):
        raise ValueError("Lần chạy này chưa có bước định vị/cắt vùng chai.")
    print(f"Evaluating {len(records)} full Test images x {len(MODEL_NAMES)} models; run {run_id}", flush=True)
    started = time.perf_counter()
    results = []
    for position, record in enumerate(records):
        # Ground truth remains outside ModelService. It receives only pixels and model ID.
        image = read_bgr(record["path"])
        for model_id in MODEL_NAMES:
            row = {
                "id": record["id"], "path": record["path"], "truth": record["label"],
                "model": model_id, "localization_rejected": False,
            }
            try:
                prediction = service.predict(image, model=model_id)
            except ValueError as error:
                if not str(error).startswith(LOCALIZATION_REJECTIONS):
                    raise RuntimeError(f"Unexpected inference error for {record['id']} / {model_id}: {error}") from error
                row.update({
                    "label": "uncertain", "raw_label": None, "confidence": None,
                    "localization_rejected": True, "reason": str(error),
                })
            else:
                if prediction.get("run_id") != run_id:
                    raise RuntimeError("latest.json đã đổi giữa lúc đánh giá; dừng để không trộn mô hình.")
                if prediction.get("label") not in ("normal", "defect", "uncertain"):
                    raise RuntimeError(f"Model returned an unsupported label: {prediction.get('label')}")
                row.update({
                    key: prediction.get(key)
                    for key in ("label", "raw_label", "confidence", "probabilities", "latency_ms", "localization", "threshold", "uncertainty_threshold")
                })
            row["correct"] = row["label"] == row["truth"]
            results.append(row)
        if position == 0 or (position + 1) % 10 == 0 or position + 1 == len(records):
            print(f"Evaluated {position + 1}/{len(records)} images ({len(results)} predictions)", flush=True)
    # Reject a changed run even when the last few images were localization rejections.
    if service.status().get("run_id") != run_id:
        raise RuntimeError("Lần huấn luyện hiện hành thay đổi trong quá trình đánh giá.")
    note = (
        "Đánh giá toàn ảnh qua định vị chai tự động và phân loại thực tế; không lưu lịch sử web. "
        "Accuracy tính đúng/tổng ảnh, uncertain và lỗi định vị không được tính đúng. "
        "Coverage là tỷ lệ nhận nhãn normal/defect. Recall lớp defect tính cả ảnh bị từ chối là bỏ sót. "
        "Test cùng nguồn đã dùng trong phát triển, chưa phải kiểm tra mù trên chai/phiên chụp độc lập."
    )
    report = {
        "status": "complete", "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(), "split": "test",
        "sample_count": len(records), "available_test_count": available,
        "partial": len(records) < available, "prediction_count": len(results),
        "unexpected_error_count": 0, "elapsed_seconds": round(time.perf_counter() - started, 4),
        "scope": manifest.get("scope", ""), "note": note,
        "models": {key: {"name": name, **summarize([row for row in results if row["model"] == key])} for key, name in MODEL_NAMES.items()},
        "records": results,
    }
    destination = Path(output) if output else run_dir / ("end_to_end.json" if not report["partial"] else f"end_to_end_sample_{len(records)}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"output": str(destination), **{key: value for key, value in report.items() if key != "records"}}, ensure_ascii=True, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=APP_DIR / "data" / "training" / "manifest.json")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        evaluate(args.manifest, args.artifact_root, args.max_samples, args.output)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
