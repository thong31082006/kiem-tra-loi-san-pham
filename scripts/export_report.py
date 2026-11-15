"""Generate a readable report from actual saved model/evaluation artifacts."""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = APP_DIR / "models" / "trained"


def percent(value):
    return f"{float(value) * 100:.2f}%".replace(".", ",")


def export_report(artifact_root=ARTIFACT_ROOT, output=None):
    artifact_root = Path(artifact_root)
    run_id = json.loads((artifact_root / "latest.json").read_text(encoding="utf-8"))["run_id"]
    run_dir = (artifact_root / run_id).resolve()
    run_dir.relative_to(artifact_root.resolve())
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    pipeline_path = run_dir / "end_to_end.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8")) if pipeline_path.is_file() else {}
    destination = Path(output) if output else APP_DIR / "BAO_CAO_MO_HINH.md"
    completed = datetime.fromisoformat(report["completed_at"]).astimezone(timezone(timedelta(hours=7)))
    lines = ["# Kết quả huấn luyện và kiểm tra chai nhựa", "",
             f"Lần chạy `{run_id}`, hoàn tất {completed:%d/%m/%Y %H:%M} (giờ Việt Nam).",
             "", "Mỗi yêu cầu dùng một ảnh. SSDLite COCO đã huấn luyện sẵn tìm chai; ba bộ phân loại chất lượng được huấn luyện bằng dữ liệu của dự án. Không dùng ảnh chuẩn hoặc so sánh với ảnh chuẩn.",
             "", "## Kết quả toàn bộ luồng web", ""]
    if pipeline:
        lines.extend([f"Đã chạy {pipeline['sample_count']} ảnh Test × 3 mô hình = {pipeline['prediction_count']} dự đoán thực tế, trong {pipeline['elapsed_seconds']:.2f} giây. Không tạo lịch sử web khi đánh giá.", "",
                      "| Mô hình | Đúng / tổng ảnh | Precision lỗi | Recall lỗi | F1 lỗi | Có kết luận | Chưa kết luận |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for entry in pipeline["models"].values():
            values = [entry["name"], *(percent(entry[key]) for key in ("accuracy", "precision", "recall", "f1", "coverage")), str(entry["uncertain_count"])]
            lines.append("| " + " | ".join(values) + " |")
        lines.extend(["", pipeline["note"], "", "Ma trận: hàng là nhãn thật; cột là nhãn được trả cho người dùng.", "",
                      "| Mô hình | Nhãn thật | Bình thường | Có lỗi | Chưa đủ tin cậy |",
                      "| --- | --- | ---: | ---: | ---: |"])
        for entry in pipeline["models"].values():
            for label, row in zip(("Bình thường", "Có lỗi"), entry["confusion_matrix"]):
                lines.append("| " + " | ".join([entry["name"], label, *map(str, row)]) + " |")
    else:
        lines.append("Chưa có báo cáo toàn bộ luồng web cho lần huấn luyện này. Chạy `scripts/evaluate_web_pipeline.py` trước khi kết luận về kết quả sử dụng thực tế.")
    lines.extend(["", "## Bộ phân loại trước bước từ chối thiếu tin cậy", "",
                  "Đầu vào cũng là vùng chai do bộ định vị tự tìm. Bảng này chưa áp dụng việc yêu cầu chụp lại khi điểm dưới 0,65.", "",
                  "| Mô hình | Accuracy | Precision lỗi | Recall lỗi | F1 lỗi | Ngưỡng phân loại |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for entry in report["models"].values():
        lines.append("| " + " | ".join([entry["name"], *(percent(entry["test"][key]) for key in ("accuracy", "precision", "recall", "f1")), f"{entry['decision_threshold']:.3f}"]) + " |")
    dataset = report["dataset"]
    svm_params = report["models"]["svm"].get("hyperparameters", {"svc__C": 2, "svc__gamma": "scale"})
    forest_params = report["models"]["random_forest"].get("hyperparameters", {"n_estimators": 160, "max_depth": 12, "min_samples_leaf": 1})
    lines.extend(["", "## Dữ liệu và cách huấn luyện", "",
                  f"- {dataset['image_count']} ảnh: {dataset['train']} Train, {dataset['val']} Validation, {dataset['test']} Test.",
                  "- Khối 40 chỉ số ảnh được giữ cùng tập; bỏ ba khung ở mỗi cạnh. Seed 42. Nhãn hiệu khác không bị coi là lỗi chất lượng.",
                  "- Tự tìm và lưu vùng chai không mất dữ liệu trước khi huấn luyện; lúc sử dụng áp dụng đúng cách xử lý này. Tên tệp, annotation và nhãn thư viện không đi vào mô hình dự đoán.",
                  f"- HOG + histogram HSV; SVM RBF: {svm_params}; Random Forest: {forest_params}. Cân bằng lớp từ dữ liệu Train.",
                  f"- MobileNetV3 Small học chuyển giao từ ImageNet; batch {report['models']['mobilenet']['batch_size']}, {report['models']['mobilenet']['epochs']} epoch trên {report['models']['mobilenet']['device']}; chọn epoch {report['models']['mobilenet']['best_epoch']} bằng Validation.",
                  f"- Chuẩn bị đặc trưng, huấn luyện và đánh giá bộ phân loại trong {report['total_seconds']:.2f} giây; chưa gồm chuẩn bị cache vùng chai và đánh giá toàn luồng web.",
                  "", "Đường học của MobileNet (F1 hai lớp trên Validation, tại ngưỡng ghi theo từng epoch):", "",
                  "| Epoch | Train loss | Validation loss | Validation F1 | Ngưỡng |",
                  "| ---: | ---: | ---: | ---: | ---: |"])
    for row in report["models"]["mobilenet"]["curves"]:
        lines.append(f"| {row['epoch']} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | {percent(row['val_f1'])} | {row.get('threshold', 0.5):.3f} |")
    audit_path = run_dir / "selection_frozen.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        lines.extend(["", "## So sánh trước / sau bằng Validation", "",
                      "Lựa chọn dưới đây đã được khóa trước khi đánh giá Test. Balanced correct tính trung bình tỷ lệ kết luận đúng của từng lớp; ảnh chưa đủ tin cậy tính là chưa kết luận đúng.", "",
                      "| Mô hình | Balanced correct trước | Sau | Báo nhầm trước | Sau | Recall lỗi trước | Sau | Trọng số được dùng |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
        for key, entry in report["models"].items():
            decision = audit["decisions"][key]
            before, after = decision["baseline"], entry["validation_pipeline"]
            state = "Huấn luyện mới" if decision["use_new_candidate"] else "Giữ bộ trước"
            lines.append("| " + " | ".join([entry["name"], percent(before["balanced_correct"]), percent(after["balanced_correct"]), str(before["false_positives"]), str(after["false_positives"]), percent(before["recall"]), percent(after["recall"]), state]) + " |")
        lines.extend(["", report["evaluation_note"], "",
                      f"Bộ trước: `{audit['baseline_run_id']}`. Manifest SHA256: `{audit['manifest_sha256']}`. Không thêm ảnh Validation/Test vào Train; ảnh hãng khác vẫn nằm ngoài phạm vi đã kiểm chứng."])
        old_pipeline_path = APP_DIR / "models" / "trained" / audit["baseline_run_id"] / "end_to_end.json"
        if pipeline and old_pipeline_path.is_file():
            old_pipeline = json.loads(old_pipeline_path.read_text(encoding="utf-8"))
            lines.extend(["", "## Đối chiếu toàn luồng Test trước / sau", "",
                          "Các lựa chọn đã khóa bằng Validation. Bảng này chỉ báo cáo kết quả trên cùng tập Test phát triển, không dùng để sửa tiếp tham số.", "",
                          "| Mô hình | Đúng / tổng trước | Sau | Báo nhầm bình thường thành lỗi (trước → sau) | Ảnh lỗi bị kết luận bình thường (trước → sau) | Chưa đủ tin cậy (trước → sau) |",
                          "| --- | ---: | ---: | ---: | ---: | ---: |"])
            for key, after in pipeline["models"].items():
                before = old_pipeline["models"][key]
                a, b = before["confusion_matrix"], after["confusion_matrix"]
                lines.append("| " + " | ".join([after["name"], percent(before["accuracy"]), percent(after["accuracy"]),
                             f"{a[0][1]} → {b[0][1]}", f"{a[1][0]} → {b[1][0]}", f"{before['uncertain_count']} → {after['uncertain_count']}"]) + " |")
            lines.extend(["", "Tỷ lệ kết luận đúng cao hơn không đồng nghĩa mọi loại sai số đều giảm. Một số ảnh trước đây chưa đủ tin cậy nay được kết luận bình thường, trong đó vẫn có ảnh lỗi. Cần xem đồng thời số báo nhầm, số ảnh lỗi bị kết luận bình thường và số ảnh chưa kết luận; chưa dùng kết quả này để bảo đảm chất lượng ngoài thực tế."])
    lines.extend(["", "## Phạm vi kết luận", "",
                  "Dữ liệu có một mẫu chai Ron 88 và bối cảnh chụp giới hạn. Nguồn không có ID chai vật lý; ảnh gần giống có thể còn xuất hiện ở các khối khác nhau. Tập Test đã được xem trong phát triển, bao gồm việc sửa chênh lệch cách cắt ảnh khi học và sử dụng. Đây là số liệu phát triển, chưa phải kiểm tra mù trên chai hoặc phiên chụp độc lập.",
                  "", "Không suy ra khả năng nhận đúng từ mọi góc chụp, ảnh ngoài nguồn hoặc lỗi bị che khuất từ các chỉ số trên. Cần bộ ảnh thật có nhiều góc và các chai/phiên chụp độc lập để đánh giá tiếp. Điểm mô hình chưa được xác nhận là xác suất đúng thực tế.",
                  "", report["latency_note"], "",
                  "Camera vật lý chưa được kiểm thử. Xem [kế hoạch thu thập dữ liệu](THU_THAP_DU_LIEU.md) và [nguồn dữ liệu](DATASET.md).", "",
                  f"Bằng chứng gốc: [cấu hình và chỉ số](models/trained/{run_id}/report.json), [toàn bộ dự đoán web](models/trained/{run_id}/end_to_end.json). Các lần chạy cũ được giữ riêng; `latest.json` xác định mô hình đang dùng.", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(export_report(args.artifact_root, args.output))
