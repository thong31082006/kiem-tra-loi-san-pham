"""Train only HOG + SVM, freeze validation choices, then test and activate.

One 6-class SVC predicts normal and five defect types. Quality uses P(defect)
= sum(P(type)) for the five defects. Test never selects C, gamma or thresholds.
"""
import argparse
import csv
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.defect_types import TYPE_CODES, raw_metrics, calibrate, accepted_metrics
from ml.features import read_model_image
from ml.hog_svm import ARTIFACT_ROOT, FEATURE_VERSION, HogSvmService, extract_hog, quality_result
from ml.train import DEFAULT_MANIFEST, Progress, _atomic_json, _choose_threshold, _metrics, load_manifest
from scripts.evaluate_web_pipeline import evaluate
from scripts.export_report import export_report


def train(manifest_path=DEFAULT_MANIFEST, artifact_root=ARTIFACT_ROOT, seed=42):
    manifest_path, artifact_root = Path(manifest_path), Path(artifact_root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = artifact_root / run_id
    run_dir.mkdir(parents=True)
    progress = Progress(artifact_root, run_id)
    started = time.perf_counter()
    try:
        progress.update("Kiểm tra dữ liệu và giữ nguyên Train / Validation / Test.")
        manifest, records, counts = load_manifest(manifest_path)
        if manifest.get("input_mode") != "auto_bottle_roi":
            raise ValueError("Chạy prepare_bottle_crops.py để dùng cùng cách cắt chai với web.")
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        splits = {split: [r for r in records if r["split"] == split] for split in ("train", "val", "test")}

        def prepare(rows):
            x, y = [], []
            for row in rows:
                defects = row.get("defect_types", [])
                if len(defects) > 1:
                    raise ValueError("Chưa hỗ trợ nhãn nhiều loại lỗi đồng thời.")
                code = defects[0] if defects else "normal"
                if code not in TYPE_CODES or (code == "normal") != (row["label"] == "normal"):
                    raise ValueError(f"Nhãn chất lượng/loại lỗi không nhất quán: {row['id']}")
                x.append(extract_hog(read_model_image(row)))
                y.append(TYPE_CODES.index(code))
            return np.asarray(x), np.asarray(y)

        x_train, y_train = prepare(splits["train"])
        x_val, y_val = prepare(splits["val"])
        if set(y_train) != set(range(6)):
            raise ValueError("Train phải đủ 6 lớp.")
        candidates, best = [], None
        for c in (2.0, 8.0):
            for gamma in ("scale", 0.00005):
                progress.update(f"Huấn luyện SVM RBF: C={c}, gamma={gamma} (chỉ fit Train).", model="svm")
                model = make_pipeline(StandardScaler(), SVC(C=c, gamma=gamma, kernel="rbf", probability=True,
                                                           class_weight="balanced", random_state=seed))
                model.fit(x_train, y_train)
                scores = model.predict_proba(x_val)
                metrics = raw_metrics(y_val, scores)
                config = {"C": c, "gamma": gamma, "kernel": "rbf", "class_weight": "balanced"}
                candidates.append({"config": config, "validation": metrics})
                rank = (metrics["macro_f1"], metrics["accuracy"])
                if best is None or rank > best[0]:
                    best = (rank, model, scores, config)
        _, model, val_scores, config = best
        threshold, val_quality = _choose_threshold((y_val != 0).astype(int), val_scores[:, 1:].sum(axis=1))
        policies = calibrate(y_val, val_scores)
        _atomic_json(run_dir / "selection_frozen.json", {
            "selection_split": "val", "manifest_sha256": manifest_hash,
            "frozen_at": datetime.now(timezone.utc).isoformat(), "candidates": candidates,
            "selected_config": config, "decision_threshold": threshold, "uncertainty_threshold": 0.65,
            "policies": policies, "selection_metric": "six-class macro_f1, then accuracy"})
        joblib.dump(model, run_dir / "svm.joblib", compress=3)
        progress.update("Đã khóa tham số và ngưỡng bằng Validation. Đang kiểm thử trên Test.", model="test")
        x_test, y_test = prepare(splits["test"])
        test_scores = model.predict_proba(x_test)
        test_quality = _metrics((y_test != 0).astype(int), test_scores[:, 1:].sum(axis=1), threshold)
        type_counts = {split: {code: sum((r.get("defect_types") or ["normal"])[0] == code for r in rows)
                               for code in TYPE_CODES} for split, rows in splits.items()}
        note = ("Chỉ fit Train; chọn C, gamma và ngưỡng bằng Validation rồi khóa trước Test. "
                "Test là đánh giá phát triển cùng nguồn, đã được tham khảo trong dự án; chưa phải kiểm chứng mù. "
                "Mỗi ảnh một chai đứng và một lỗi chính. Điểm chưa được hiệu chuẩn.")
        report = {"status": "complete", "run_id": run_id, "seed": seed, "input_mode": "auto_bottle_roi",
                  "class_names": list(TYPE_CODES), "feature_version": FEATURE_VERSION,
                  "feature_count": 3780, "policies": policies, "evaluation_note": note,
                  "dataset": {"image_count": len(records), **{s: len(r) for s, r in splits.items()}, "counts": counts},
                  "manifest_sha256": manifest_hash, "models": {"svm": {
                      "name": "HOG + SVM", "artifact": "svm.joblib", "hyperparameters": config,
                      "decision_threshold": threshold, "uncertainty_threshold": 0.65,
                      "validation": val_quality, "test": test_quality}},
                  "defect_types": {"counts": type_counts, "test": {"raw": raw_metrics(y_test, test_scores),
                                                          "accepted": accepted_metrics(y_test, test_scores, policies)}}}
        _atomic_json(run_dir / "report.json", report)
        with (run_dir / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "truth_quality", "truth_type", "predicted_quality", "predicted_type_raw", "confidence"])
            for row, scores in zip(splits["test"], test_scores):
                result = quality_result(scores, threshold)
                writer.writerow([row["id"], row["label"], (row.get("defect_types") or ["normal"])[0],
                                 result["label"], TYPE_CODES[int(scores.argmax())], result["confidence"]])
        progress.update("Đang test toàn luồng: ảnh gốc → tìm chai → HOG → SVM.", model="test")
        evaluate(manifest_path, artifact_root, service=HogSvmService(artifact_root, run_id=run_id))
        report["total_seconds"] = round(time.perf_counter() - started, 2)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(run_dir / "report.json", report)
        export_report(artifact_root, output=run_dir / "summary.md", run_id=run_id)
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_hash:
            raise RuntimeError("Manifest đã đổi; giữ nguyên mô hình đang dùng.")
        # Publish only after training AND full pipeline testing succeeded.
        _atomic_json(artifact_root / "latest.json", {"run_id": run_id})
        progress.update("Đã huấn luyện và test HOG + SVM thành công.", state="complete", model="svm")
        return report
    except BaseException as error:
        progress.update(f"Tác vụ dừng: {error}", state="failed")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    train(args.manifest, args.artifact_root, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
