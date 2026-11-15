"""Train candidates, freeze Validation-only decisions, evaluate once, then activate.

The existing dataset/splits and production artifacts remain intact. Test metrics
never choose a candidate or its threshold. Non-improving models retain their old
weights and thresholds. Every selection decision is saved before reading Test.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.features import APP_DIR, CLASS_NAMES, FEATURE_VERSION, cnn_transform
from ml.inference import MODEL_NAMES
from ml.selection import accept_candidate, choose_pipeline_threshold, pipeline_metrics
from ml.train import (
    DEFAULT_ARTIFACT_ROOT, DEFAULT_MANIFEST, UNCERTAINTY_THRESHOLD, Progress,
    _atomic_json, _classical_latency, _defect_probabilities, _metrics, _prepare,
    _train_cnn, load_manifest,
)
from scripts.evaluate_web_pipeline import evaluate
from scripts.export_report import export_report


class DualProgress(Progress):
    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        _atomic_json(DEFAULT_ARTIFACT_ROOT / "training_progress.json", json.loads(self.path.read_text(encoding="utf-8")))


def probabilities(model_id, entry, run_dir, features, images, positions):
    artifact = run_dir / entry["artifact"]
    if model_id != "mobilenet":
        return _defect_probabilities(joblib.load(artifact), features[positions])
    import torch
    from PIL import Image
    from torchvision.models import mobilenet_v3_small

    network = mobilenet_v3_small(weights=None)
    network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, 2)
    network.load_state_dict(torch.load(artifact, map_location="cpu", weights_only=True)["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device).eval()
    transform = cnn_transform()
    values = []
    with torch.inference_mode():
        for start in range(0, len(positions), 8):
            batch = torch.stack([transform(Image.fromarray(images[index])) for index in positions[start:start + 8]]).to(device)
            values.extend(network(batch).softmax(dim=1)[:, 1].cpu().tolist())
    del network
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(values)


def train_classical(model_id, features, labels, indices, records, run_dir, progress, seed):
    train = indices["train"]
    groups = np.asarray([records[index]["group"] for index in train])
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(features[train], labels[train], groups))
    for fit, check in folds:
        assert not set(groups[fit]) & set(groups[check])
        assert len(np.unique(labels[train][fit])) == 2
        assert len(np.unique(labels[train][check])) == 2
    if model_id == "svm":
        base = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", random_state=seed))
        grid = {"svc__C": [0.5, 2.0, 8.0, 32.0], "svc__gamma": ["scale", 0.00005, 0.001]}
    else:
        base = RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=2)
        grid = {"n_estimators": [160, 320], "max_depth": [8, 12, None], "min_samples_leaf": [1, 3]}
    trials = []
    best = None
    started = time.perf_counter()
    parameters = list(ParameterGrid(grid))
    for index, params in enumerate(parameters):
        progress.update(f"{MODEL_NAMES[model_id]}: chọn cấu hình {index + 1}/{len(parameters)} bằng 3-fold theo nhóm trong Train", model=model_id)
        scores = cross_val_score(clone(base).set_params(**params), features[train], labels[train], cv=folds,
                                 scoring="balanced_accuracy", n_jobs=1, error_score="raise")
        trial = {"params": params, "fold_scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std())}
        trials.append(trial)
        rank = (trial["mean"], -trial["std"])
        if best is None or rank > best[0]:
            best = (rank, params)
    model = clone(base).set_params(**best[1])
    if model_id == "svm":
        model.set_params(svc__probability=True)
    progress.update(f"Đang fit {MODEL_NAMES[model_id]} trên toàn bộ Train; tham số {best[1]}", model=model_id)
    model.fit(features[train], labels[train])
    val_probabilities = _defect_probabilities(model, features[indices["val"]])
    threshold, summary = choose_pipeline_threshold(labels[indices["val"]], val_probabilities)
    artifact = f"{model_id}.joblib"
    joblib.dump(model, run_dir / artifact, compress=3)
    return {
        "name": MODEL_NAMES[model_id], "artifact": artifact, "test": {},
        "validation": _metrics(labels[indices["val"]], val_probabilities, threshold),
        "validation_pipeline": summary, "decision_threshold": threshold,
        "uncertainty_threshold": UNCERTAINTY_THRESHOLD, "feature_version": FEATURE_VERSION,
        "training_seconds": round(time.perf_counter() - started, 4),
        "inference_ms": round(_classical_latency(model, [records[index] for index in indices["val"]]), 4),
        "curves": [], "hyperparameters": best[1],
        "group_cv": {"split": "train", "folds": 3, "scoring": "balanced_accuracy", "trials": trials},
    }


def retrain(epochs=20, batch_size=8, seed=42, activate=False):
    started = time.perf_counter()
    baseline_pointer = json.loads((DEFAULT_ARTIFACT_ROOT / "latest.json").read_text(encoding="utf-8"))
    baseline_id = baseline_pointer["run_id"]
    baseline_dir = DEFAULT_ARTIFACT_ROOT / baseline_id
    baseline_report = json.loads((baseline_dir / "report.json").read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    candidate_root = APP_DIR / "models" / "retraining_candidates"
    run_dir = candidate_root / run_id
    run_dir.mkdir(parents=True)
    progress = DualProgress(candidate_root, run_id)
    try:
        random.seed(seed)
        np.random.seed(seed)
        cv2.setNumThreads(2)
        progress.update("Kiểm tra dữ liệu và giữ nguyên Train / Validation / Test", model="preparation")
        manifest, records, counts = load_manifest(DEFAULT_MANIFEST)
        if manifest.get("input_mode") != "auto_bottle_roi":
            raise ValueError("Cần cùng vùng chai tự động như web.")
        manifest_hash = hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()
        features, images, labels, indices = _prepare(records, progress)
        baseline_summaries = {}
        for model_id, entry in baseline_report["models"].items():
            values = probabilities(model_id, entry, baseline_dir, features, images, indices["val"])
            baseline_summaries[model_id] = pipeline_metrics(labels[indices["val"]], values, entry["decision_threshold"], entry.get("uncertainty_threshold", 0.65))
        _atomic_json(run_dir / "baseline_validation.json", baseline_summaries)
        candidates = {}
        for model_id in ("svm", "random_forest"):
            candidates[model_id] = train_classical(model_id, features, labels, indices, records, run_dir, progress, seed)
        candidates["mobilenet"] = _train_cnn(
            images, labels, indices, records, run_dir, progress, epochs, batch_size, "auto", seed,
            selection="balanced_pipeline", finetune_blocks=4, feature_lr=3e-5,
            freeze_batchnorm=True, evaluate_test=False,
        )
        selected, decisions = {}, {}
        for model_id, entry in candidates.items():
            accepted = accept_candidate(entry["validation_pipeline"], baseline_summaries[model_id])
            decisions[model_id] = {
                "use_new_candidate": accepted, "baseline": baseline_summaries[model_id],
                "candidate": entry["validation_pipeline"], "candidate_config": entry,
            }
            if accepted:
                selected[model_id] = entry
                entry["provenance"] = {"trained_in_run": run_id, "selection_split": "val"}
            else:
                attempt = run_dir / ("attempt_" + entry["artifact"])
                (run_dir / entry["artifact"]).rename(attempt)
                decisions[model_id]["attempt_artifact"] = attempt.name
                selected[model_id] = copy.deepcopy(baseline_report["models"][model_id])
                shutil.copy2(baseline_dir / selected[model_id]["artifact"], run_dir / selected[model_id]["artifact"])
                selected[model_id]["validation_pipeline"] = baseline_summaries[model_id]
                selected[model_id]["provenance"] = {"retained_from_run": baseline_id, "selection_split": "val"}
        decision = {
            "frozen_at": datetime.now(timezone.utc).isoformat(), "selection_split": "val",
            "baseline_run_id": baseline_id, "manifest_sha256": manifest_hash,
            "policy": "Higher balanced per-class correct rate including abstentions as failures; false positives must not rise and defect recall may drop at most 2 percentage points. Thresholds use Validation only; hyperparameters use grouped CV inside Train only.",
            "decisions": decisions,
        }
        # This file is frozen BEFORE any Test probabilities are evaluated.
        _atomic_json(run_dir / "selection_frozen.json", decision)
        for model_id, entry in selected.items():
            progress.update(f"Đã khóa lựa chọn bằng Validation. Đang đánh giá Test: {MODEL_NAMES[model_id]}", model="evaluation")
            values = probabilities(model_id, entry, run_dir, features, images, indices["test"])
            entry["test"] = _metrics(labels[indices["test"]], values, entry["decision_threshold"])
        report = {
            "schema_version": 1, "status": "complete", "run_id": run_id, "seed": seed,
            "completed_at": datetime.now(timezone.utc).isoformat(), "input_mode": "auto_bottle_roi",
            "models": selected, "class_names": list(CLASS_NAMES), "feature_version": FEATURE_VERSION,
            "total_seconds": round(time.perf_counter() - started, 4),
            "dataset": {**baseline_report["dataset"], "manifest_sha256": manifest_hash},
            "evaluation_note": "Cấu hình chọn bằng cross-validation theo nhóm trong Train; epoch, ngưỡng và quyết định giữ/thay mô hình chỉ dùng Validation. Test chỉ báo cáo sau khi khóa lựa chọn. Tập Test từng được xem trong phát triển, chưa phải kiểm tra mù độc lập. Điểm mô hình chưa hiệu chuẩn.",
            "latency_note": baseline_report["latency_note"], "selection_audit": "selection_frozen.json",
            "retraining": {"baseline_run_id": baseline_id, "new_models": [key for key, value in decisions.items() if value["use_new_candidate"]], "attempted_epochs": epochs},
        }
        _atomic_json(run_dir / "report.json", report)
        _atomic_json(candidate_root / "latest.json", {"run_id": run_id})
        progress.update("Đang đánh giá toàn bộ luồng web trên 231 ảnh Test, không ghi lịch sử", model="evaluation")
        evaluate(DEFAULT_MANIFEST, candidate_root)
        export_report(candidate_root, run_dir / "summary.md")
        activated = False
        if activate and report["retraining"]["new_models"]:
            current = json.loads((DEFAULT_ARTIFACT_ROOT / "latest.json").read_text(encoding="utf-8"))
            if current != baseline_pointer:
                raise RuntimeError("Bộ mô hình hiện hành đã đổi trong lúc huấn luyện; giữ kết quả ứng viên để tránh ghi đè.")
            if hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest() != manifest_hash:
                raise RuntimeError("Manifest thay đổi trong lúc huấn luyện; không kích hoạt.")
            destination = DEFAULT_ARTIFACT_ROOT / run_id
            shutil.copytree(run_dir, destination)
            _atomic_json(DEFAULT_ARTIFACT_ROOT / "latest.json", {"run_id": run_id})
            activated = True
            export_report(DEFAULT_ARTIFACT_ROOT)
        outcome = {"run_id": run_id, "baseline_run_id": baseline_id, "activated": activated,
                   "new_models": report["retraining"]["new_models"], "report": str(run_dir / "summary.md")}
        _atomic_json(run_dir / "outcome.json", outcome)
        progress.update("Đã hoàn tất huấn luyện, đánh giá và chuyển sang bộ đã chọn bằng Validation." if activated else "Đã hoàn tất; giữ bộ hiện hành vì chưa có ứng viên vượt điều kiện Validation.",
                        state="complete", **outcome, total_seconds=round(time.perf_counter() - started, 2))
        print(json.dumps(outcome, ensure_ascii=False), flush=True)
        return outcome
    except BaseException as error:
        progress.update(f"Huấn luyện dừng; giữ nguyên mô hình đang dùng: {error}", state="failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 30 or not 1 <= args.batch_size <= 16:
        parser.error("epochs: 1..30; batch-size: 1..16")
    try:
        retrain(args.epochs, args.batch_size, args.seed, args.activate)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
