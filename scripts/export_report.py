"""Export the active HOG + SVM report; no model comparison or retraining."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.defect_types import TYPE_CODES, TYPE_NAMES
from ml.features import APP_DIR
from ml.hog_svm import ARTIFACT_ROOT, FEATURE_VERSION


def export_report(artifact_root=ARTIFACT_ROOT, output=None, run_id=None):
    root = Path(artifact_root).resolve()
    if run_id is None:
        run_id = json.loads((root / "latest.json").read_text(encoding="utf-8"))["run_id"]
    run_dir = (root / run_id).resolve()
    run_dir.relative_to(root)
    if run_dir == root:
        raise ValueError("Đường dẫn báo cáo không hợp lệ.")
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Chưa có báo cáo HOG + SVM hoàn tất.")
    entry = report["models"]["svm"]
    dataset = report["dataset"]
    lines = ["# Kết quả HOG + SVM — Nhóm 8", "", report["evaluation_note"], "",
             f"Train / Validation / Test: **{dataset['train']} / {dataset['val']} / {dataset['test']}** ảnh.",
             "HOG ảnh xám 64 × 128: cell 8 × 8, block 16 × 16, bước 8 × 8, 9 hướng; **3.780 đặc trưng**.",
             f"StandardScaler → SVC RBF: `{entry['hyperparameters']}`.",
             "Một SVM sáu lớp cho cả kết luận chất lượng và tên lỗi; chất lượng gộp năm lớp lỗi.",
             f"Ngưỡng chất lượng: {entry['decision_threshold']:.3f}; ngưỡng từ chối: {entry['uncertainty_threshold']:.2f}.", "",
             "## Test chất lượng", "",
             "| Đánh giá | Accuracy | Precision lỗi | Recall lỗi | F1 lỗi |",
             "| --- | ---: | ---: | ---: | ---: |"]

    def add_metrics(label, metrics):
        lines.append("| " + label + " | " + " | ".join(f"{metrics[key]:.2%}" for key in
                     ("accuracy", "precision", "recall", "f1")) + " |")

    add_metrics("Phân loại vùng chai, trước bước từ chối", entry["test"])
    pipeline_path = run_dir / "end_to_end.json"
    if pipeline_path.is_file():
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        metrics = pipeline["models"]["svm"]
        add_metrics("Toàn luồng, gồm ảnh chưa kết luận", metrics)
        lines.extend(["", f"Toàn luồng: đúng **{metrics['correct_count']}/{metrics['total']}** ảnh; "
                      f"{metrics['uncertain_count']} ảnh chưa kết luận; tỷ lệ có kết luận {metrics['coverage']:.2%}.",
                      "Accuracy toàn luồng tính ảnh chưa kết luận là chưa đúng. Precision/Recall/F1 dùng lỗi làm lớp dương.",
                      "", "### Ma trận nhầm lẫn toàn luồng", "",
                      "| Nhãn thật | Đạt | Không đạt | Chưa kết luận |", "| --- | ---: | ---: | ---: |"])
        for label, values in zip(("Bình thường", "Có lỗi"), metrics["confusion_matrix"]):
            lines.append("| " + label + " | " + " | ".join(map(str, values)) + " |")
    types = report["defect_types"]
    lines.extend(["", "## Gọi tên lỗi từ cùng SVM", "",
                  "Ngưỡng gọi tên khóa bằng Validation: Precision ≥ 85%, Recall ≥ 20%, ít nhất 3 phát hiện đúng.",
                  "Lớp không đạt điều kiện không được gọi tên, dù có điểm thô cao.", "",
                  "| Loại lỗi | Ảnh Train | Ảnh Test | Cho phép gọi tên | Precision Test | Recall Test | F1 Test |",
                  "| --- | ---: | ---: | --- | ---: | ---: | ---: |"])
    for code in TYPE_CODES[1:]:
        values = types["test"]["accepted"]["per_class"][code]
        enabled = "Có" if report["policies"][code]["enabled"] else "Chưa đủ tin cậy"
        lines.append(f"| {TYPE_NAMES[code]} | {types['counts']['train'][code]} | {types['counts']['test'][code]} | {enabled} | "
                     + " | ".join(f"{values[key]:.2%}" for key in ("precision", "recall", "f1")) + " |")
    lines.extend(["", "## Phạm vi và cách chạy lại", "",
                  "SSDLite có sẵn chỉ tìm/cắt chai trước HOG; không phân loại chất lượng hay loại lỗi.",
                  "Cần ảnh một chai đứng. Chưa hỗ trợ móp chai, nước đục, lỗi bị che khuất và nhiều lỗi đồng thời.",
                  "Nhãn hỏng/lệch/thiếu là một nhóm chung theo annotation nguồn.",
                  "Test cùng nguồn đã được tham khảo trong phát triển, chưa phải kiểm chứng mù trên chai/phiên chụp mới.", "",
                  "Train: `python scripts/train_hog_svm.py`. Test độc lập: `python scripts/test_hog_svm.py`.",
                  "Chi tiết: `report.json`, `test_predictions.csv`, `end_to_end.json` trong thư mục lần chạy.",
                  f"Run: `{run_id}`. Manifest SHA256: `{report['manifest_sha256']}`."])
    destination = Path(output) if output else APP_DIR / "BAO_CAO_MO_HINH.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(export_report(args.artifact_root, args.output))
