"""Train six-class candidates on Train, select/calibrate on Val, report Test once."""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.defect_types import TYPE_CODES, TYPE_NAMES, TYPE_ROOT, TYPE_SCOPE, classify_type_scores
from ml.features import APP_DIR, IMAGE_SIZE, as_rgb_pil, cnn_transform, extract_features, read_model_image
from ml.train import DEFAULT_MANIFEST, _atomic_json, load_manifest


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


def main(epochs, batch_size):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4); cv2.setNumThreads(2)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = TYPE_ROOT / run_id
    run_dir.mkdir(parents=True)
    started = time.perf_counter()

    def progress(message, **extra):
        print(message, flush=True)
        _atomic_json(TYPE_ROOT / "training_progress.json", {"state": "running", "run_id": run_id, "message": message, **extra})

    try:
        progress("Kiểm tra nhãn loại lỗi, giữ nguyên split của dự án")
        manifest, records, _ = load_manifest(DEFAULT_MANIFEST)
        manifest_hash = hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()
        labels, features, thumbnails = [], [], []
        positions = {key: [] for key in ("train", "val", "test")}
        for i, record in enumerate(records):
            defects = record.get("defect_types", [])
            if len(defects) > 1:
                raise ValueError("Có ảnh nhiều loại lỗi; cần chuyển sang bài toán đa nhãn trước khi huấn luyện.")
            code = defects[0] if defects else "normal"
            if code not in TYPE_CODES or (record["label"] == "normal") != (code == "normal"):
                raise ValueError("Nhãn chất lượng và loại lỗi không nhất quán.")
            image = read_model_image(record)
            features.append(extract_features(image))
            thumbnails.append(np.asarray(as_rgb_pil(image).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)).copy())
            labels.append(TYPE_CODES.index(code)); positions[record["split"]].append(i)
        x, y = np.asarray(features), np.asarray(labels)
        train, val, test = (positions[key] for key in ("train", "val", "test"))
        counts = {split: {code: int((y[indices] == i).sum()) for i, code in enumerate(TYPE_CODES)} for split, indices in positions.items()}
        assert all(min(values.values()) > 0 for values in counts.values())
        candidates = []
        for c in (2.0, 8.0):
            for gamma in ("scale", 0.00005):
                progress(f"Huấn luyện SVM nhận diện 6 nhóm: C={c}, gamma={gamma}")
                model = make_pipeline(StandardScaler(), SVC(C=c, gamma=gamma, probability=True, class_weight="balanced", random_state=seed))
                model.fit(x[train], y[train])
                scores = model.predict_proba(x[val])
                artifact = f"svm_{len(candidates)}.joblib"
                joblib.dump(model, run_dir / artifact, compress=3)
                candidates.append({"model_id": "svm", "model_name": "HOG + SVM · phân tích 6 nhóm", "artifact": artifact,
                                   "validation": raw_metrics(y[val], scores), "validation_scores": scores,
                                   "config": {"C": c, "gamma": gamma}})
        for trees in (160, 320):
            progress(f"Huấn luyện Random Forest nhận diện 6 nhóm: {trees} cây")
            model = RandomForestClassifier(n_estimators=trees, max_depth=12, class_weight="balanced", random_state=seed, n_jobs=2)
            model.fit(x[train], y[train])
            scores = model.predict_proba(x[val])
            artifact = f"forest_{trees}.joblib"
            joblib.dump(model, run_dir / artifact, compress=3)
            candidates.append({"model_id": "random_forest", "model_name": "HOG + Random Forest · phân tích 6 nhóm", "artifact": artifact,
                               "validation": raw_metrics(y[val], scores), "validation_scores": scores,
                               "config": {"n_estimators": trees, "max_depth": 12}})

        class Images(Dataset):
            def __init__(self, indices, augment=False):
                self.indices, self.transform = indices, cnn_transform(augment)
            def __len__(self): return len(self.indices)
            def __getitem__(self, index):
                i = self.indices[index]
                return self.transform(Image.fromarray(thumbnails[i])), int(y[i])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.hub.set_dir(str(APP_DIR / "models/pretrained"))
        network = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        for parameter in network.parameters(): parameter.requires_grad = False
        network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, len(TYPE_CODES))
        for parameter in network.classifier.parameters(): parameter.requires_grad = True
        network.to(device)
        loader = DataLoader(Images(train, True), batch_size=batch_size, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(Images(val), batch_size=batch_size, shuffle=False, num_workers=0)
        weights = len(train) / (len(TYPE_CODES) * np.bincount(y[train], minlength=len(TYPE_CODES)))
        loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(network.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

        def nn_scores(indices):
            network.eval()
            values = []
            with torch.inference_mode():
                for inputs, _ in DataLoader(Images(indices), batch_size=batch_size, shuffle=False):
                    values.extend(network(inputs.to(device)).softmax(dim=1).cpu().tolist())
            return np.asarray(values)

        best, curves = None, []
        for epoch in range(epochs):
            if epoch == 2:
                adapted = list(network.features.children())[-4:]
                parameters = []
                for block in adapted:
                    for parameter in block.parameters():
                        parameter.requires_grad = True; parameters.append(parameter)
                optimizer.add_param_group({"params": parameters, "lr": 3e-5})
            if epoch in (10, 17):
                for group in optimizer.param_groups: group["lr"] *= 0.3
            network.train(); network.features.eval()
            total_loss = 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(network(inputs), targets)
                loss.backward(); optimizer.step()
                total_loss += loss.item() * len(targets)
            scores = nn_scores(val)
            metrics = raw_metrics(y[val], scores)
            curves.append({"epoch": epoch + 1, "loss": total_loss / len(train), "validation_macro_f1": metrics["macro_f1"]})
            rank = (metrics["macro_f1"], metrics["accuracy"], -curves[-1]["loss"])
            if best is None or rank > best[0]:
                best = (rank, {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}, epoch + 1, scores.copy(), metrics)
            progress(f"MobileNet nhận diện loại lỗi: epoch {epoch + 1}/{epochs}, macro-F1 Validation={metrics['macro_f1']:.4f}", epoch=epoch + 1, epochs=epochs)
        network.load_state_dict(best[1])
        torch.save({"state_dict": best[1], "class_names": list(TYPE_CODES), "best_epoch": best[2]}, run_dir / "mobilenet.pt")
        candidates.append({"model_id": "mobilenet", "model_name": "MobileNetV3 Small · phân tích 6 nhóm", "artifact": "mobilenet.pt",
                           "validation": best[4], "validation_scores": best[3],
                           "config": {"epochs": epochs, "best_epoch": best[2], "batch_size": batch_size, "device": str(device), "curves": curves}})
        selected = max(candidates, key=lambda item: (item["validation"]["macro_f1"], item["validation"]["accuracy"]))
        policies = calibrate(y[val], selected["validation_scores"])
        audit = {"selection_split": "val", "frozen_at": datetime.now(timezone.utc).isoformat(), "manifest_sha256": manifest_hash,
                 "selected_artifact": selected["artifact"], "policies": policies,
                 "candidates": [{key: value for key, value in item.items() if key != "validation_scores"} for item in candidates]}
        _atomic_json(run_dir / "selection_frozen.json", audit)
        progress(f"Đã khóa lựa chọn bằng Validation: {selected['model_name']}. Đang đánh giá Test.")
        if selected["model_id"] == "mobilenet":
            scores = nn_scores(test)
        else:
            scores = joblib.load(run_dir / selected["artifact"]).predict_proba(x[test])
        report = {"status": "complete", "run_id": run_id, "class_names": list(TYPE_CODES),
                  **{key: value for key, value in selected.items() if key != "validation_scores"},
                  "policies": policies, "scope": TYPE_SCOPE, "counts": counts,
                  "manifest_sha256": manifest_hash, "completed_at": datetime.now(timezone.utc).isoformat(),
                  "validation_accepted": accepted_metrics(y[val], selected["validation_scores"], policies),
                  "test": {"raw": raw_metrics(y[test], scores), "accepted": accepted_metrics(y[test], scores, policies)},
                  "elapsed_seconds": round(time.perf_counter() - started, 2),
                  "note": "Nhãn loại lỗi chỉ dùng làm mục tiêu huấn luyện/đánh giá, không đưa vào dự đoán. Chọn cấu hình và ngưỡng bằng Validation trước khi xem Test; Test là đánh giá phát triển. Mỗi ảnh nguồn có một loại lỗi chính; chưa đánh giá nhiều lỗi đồng thời."}
        _atomic_json(run_dir / "report.json", report)
        _atomic_json(run_dir / "test_predictions.json", [{"id": records[index]["id"], "truth": TYPE_CODES[y[index]],
                     **classify_type_scores(row, policies)} for index, row in zip(test, scores)])
        lines = ["# Bộ AI nhận diện loại lỗi", "", selected["model_name"], "", TYPE_SCOPE, "", report["note"], "",
                 "| Loại | Train | Validation | Test | Precision Test | Recall Test | F1 Test | Bật nhận diện |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for code in TYPE_CODES:
            metrics = report["test"]["accepted"]["per_class"][code]
            lines.append("| " + " | ".join([TYPE_NAMES[code], *(str(counts[split][code]) for split in ("train", "val", "test")),
                         *(f"{metrics[key]*100:.2f}%" for key in ("precision", "recall", "f1")), "Có" if policies[code]["enabled"] else "Chưa đạt điều kiện Validation"]) + " |")
        lines.extend(["", "Nhãn hỏng/lệch/thiếu là một nhóm chung theo annotation nguồn. Chưa hỗ trợ móp chai và nước đục. Không suy ra lực siết hoặc độ kín của nắp chỉ từ ảnh.",
                      "", f"Run: `{run_id}`; manifest SHA256: `{manifest_hash}`. Quyết định trước Test lưu tại `selection_frozen.json`."])
        (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        if not any(policies[code]["enabled"] for code in TYPE_CODES[1:]):
            raise RuntimeError("Chưa loại lỗi nào đạt điều kiện Validation; không kích hoạt.")
        if hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest() != manifest_hash:
            raise RuntimeError("Manifest thay đổi; không kích hoạt.")
        _atomic_json(TYPE_ROOT / "latest.json", {"run_id": run_id})
        _atomic_json(TYPE_ROOT / "training_progress.json", {"state": "complete", "run_id": run_id, "message": "Đã huấn luyện và đánh giá bộ phân tích loại lỗi."})
        print(json.dumps({"run_id": run_id, "model": selected["model_name"], "test": report["test"], "policies": policies}, ensure_ascii=False), flush=True)
    except BaseException as error:
        _atomic_json(TYPE_ROOT / "training_progress.json", {"state": "failed", "run_id": run_id, "message": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 40 or not 1 <= args.batch_size <= 16:
        parser.error("epochs 1..40; batch-size 1..16")
    main(args.epochs, args.batch_size)
