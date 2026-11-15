"""Reproducible pixel-only training; validation selects models, test only reports."""
import argparse
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .features import (
    APP_DIR, CLASS_NAMES, FEATURE_VERSION, IMAGE_SIZE,
    as_rgb_pil, cnn_transform, extract_features, image_path, read_bgr, read_model_image,
)

DEFAULT_MANIFEST = APP_DIR / "data" / "training" / "manifest.json"
DEFAULT_ARTIFACT_ROOT = APP_DIR / "models" / "trained"
UNCERTAINTY_THRESHOLD = 0.65


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05)


class Progress:
    def __init__(self, artifact_root, run_id):
        self.path = Path(artifact_root) / "training_progress.json"
        self.run_id = run_id
        self.started_at = datetime.now(timezone.utc).isoformat()

    def update(self, message, state="running", model=None, **values):
        payload = {
            "state": state, "running": state == "running", "run_id": self.run_id,
            "started_at": self.started_at, "updated_at": datetime.now(timezone.utc).isoformat(),
            "model": model, "message": message, **values,
        }
        _atomic_json(self.path, payload)
        print(message, flush=True)


def load_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest chưa có records để huấn luyện.")
    seen_ids, seen_paths, group_splits = set(), set(), {}
    counts = {split: {label: 0 for label in CLASS_NAMES} for split in ("train", "val", "test")}
    for row in records:
        if row.get("label") not in CLASS_NAMES or row.get("split") not in counts:
            raise ValueError(f"Nhãn hoặc split không hợp lệ: {row.get('id')}")
        record_id = str(row.get("id", ""))
        if not record_id or record_id in seen_ids:
            raise ValueError("Mỗi ảnh trong manifest cần một id duy nhất.")
        seen_ids.add(record_id)
        path_value = image_path(row["path"]).resolve()
        path_value.relative_to(APP_DIR.resolve())
        if not path_value.is_file():
            raise FileNotFoundError(f"Không tìm thấy ảnh huấn luyện: {path_value}")
        normalized_path = os.path.normcase(str(path_value))
        if normalized_path in seen_paths:
            raise ValueError(f"Ảnh bị lặp trong manifest: {row['path']}")
        seen_paths.add(normalized_path)
        if manifest.get("input_mode") == "auto_bottle_roi" and not row.get("training_path"):
            raise ValueError(f"Ảnh chưa được chuẩn bị vùng chai tự động: {record_id}")
        if row.get("training_path"):
            training_path = image_path(row["training_path"]).resolve()
            training_path.relative_to(APP_DIR.resolve())
            if not training_path.is_file():
                raise FileNotFoundError(f"Không tìm thấy vùng chai đã chuẩn bị: {training_path}")
        group = str(row.get("group", ""))
        if not group:
            raise ValueError(f"Ảnh chưa có group chống rò rỉ dữ liệu: {record_id}")
        previous_split = group_splits.setdefault(group, row["split"])
        if previous_split != row["split"]:
            raise ValueError(f"Group xuất hiện trong nhiều split: {group}")
        counts[row["split"]][row["label"]] += 1
    for split, labels in counts.items():
        if any(count < (2 if split == "train" else 1) for count in labels.values()):
            raise ValueError(f"Tập {split} cần có đủ ảnh của cả hai lớp normal/defect: {labels}")
    return manifest, records, counts


def _metrics(labels, defect_probabilities, threshold):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    predictions = (np.asarray(defect_probabilities) >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "class_order": list(CLASS_NAMES),
        "positive_class": "defect", "sample_count": int(len(labels)),
    }


def _choose_threshold(labels, probabilities):
    # Every candidate is assessed on validation only. Prefer a central threshold on ties.
    best = None
    for threshold in np.linspace(0.1, 0.9, 33):
        scores = _metrics(labels, probabilities, float(threshold))
        rank = (scores["f1"], scores["balanced_accuracy"], -abs(float(threshold) - 0.5))
        if best is None or rank > best[0]:
            best = (rank, float(threshold), scores)
    return best[1], best[2]


def _prepare(records, progress):
    from PIL import Image

    features, images, labels = [], [], []
    indices = {split: [] for split in ("train", "val", "test")}
    for index, row in enumerate(records):
        image = read_model_image(row)
        features.append(extract_features(image))
        # Cache uint8 RGB thumbnails, not the original multi-megapixel images or float tensors.
        small = as_rgb_pil(image).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        images.append(np.asarray(small, dtype=np.uint8).copy())
        labels.append(CLASS_NAMES.index(row["label"]))
        indices[row["split"]].append(index)
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(records):
            progress.update(f"Đang đọc ảnh và tính đặc trưng: {index + 1}/{len(records)}", model="preparation", images_done=index + 1, images_total=len(records))
    return np.asarray(features, dtype=np.float32), images, np.asarray(labels, dtype=np.int64), indices


def _defect_probabilities(model, features):
    classes = list(model.classes_)
    return model.predict_proba(features)[:, classes.index(1)]


def _classical_latency(model, validation_records):
    durations = []
    for row in validation_records[:10]:
        image = read_model_image(row)
        started = time.perf_counter()
        model.predict_proba(extract_features(image).reshape(1, -1))
        durations.append((time.perf_counter() - started) * 1000)
    return float(np.mean(durations))


def _train_classical(model_id, model, features, labels, indices, records, run_dir, progress):
    import joblib

    names = {"svm": "HOG + SVM", "random_forest": "HOG + Random Forest"}
    progress.update(f"Đang huấn luyện {names[model_id]}", model=model_id)
    started = time.perf_counter()
    model.fit(features[indices["train"]], labels[indices["train"]])
    training_seconds = time.perf_counter() - started
    validation_probabilities = _defect_probabilities(model, features[indices["val"]])
    threshold, validation = _choose_threshold(labels[indices["val"]], validation_probabilities)
    # Test is measured once, after the threshold has been fixed using validation.
    test_probabilities = _defect_probabilities(model, features[indices["test"]])
    test = _metrics(labels[indices["test"]], test_probabilities, threshold)
    artifact = f"{model_id}.joblib"
    joblib.dump(model, run_dir / artifact, compress=3)
    inference_ms = _classical_latency(model, [records[index] for index in indices["val"]])
    progress.update(f"Đã hoàn tất {names[model_id]}; validation F1={validation['f1']:.4f}", model=model_id)
    return {
        "name": names[model_id], "artifact": artifact, "test": test,
        "validation": validation, "decision_threshold": threshold,
        "uncertainty_threshold": UNCERTAINTY_THRESHOLD,
        "training_seconds": round(training_seconds, 4), "inference_ms": round(inference_ms, 4),
        "feature_version": FEATURE_VERSION, "curves": [],
    }


def _cnn_latency(network, transform, device, validation_records, torch):
    durations = []
    network.eval()
    for row in validation_records[:10]:
        image = read_model_image(row)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        tensor = transform(as_rgb_pil(image)).unsqueeze(0).to(device)
        with torch.inference_mode():
            network(tensor).softmax(dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append((time.perf_counter() - started) * 1000)
    return float(np.mean(durations))


def _train_cnn(images, labels, indices, records, run_dir, progress, epochs, batch_size, device_name, seed,
               selection="f1", finetune_blocks=2, feature_lr=1e-5, freeze_batchnorm=False,
               evaluate_test=True):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã chọn CUDA nhưng PyTorch không tìm thấy GPU CUDA.")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.hub.set_dir(str(APP_DIR / "models" / "pretrained"))
    progress.update(f"Đang nạp MobileNetV3 Small pretrained trên {device}", model="mobilenet", epochs=epochs, device=str(device))
    network = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    for parameter in network.parameters():
        parameter.requires_grad = False
    network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, len(CLASS_NAMES))
    for parameter in network.classifier.parameters():
        parameter.requires_grad = True
    network.to(device)

    class CachedImages(Dataset):
        def __init__(self, positions, augment=False):
            self.positions = positions
            self.transform = cnn_transform(augment=augment)

        def __len__(self):
            return len(self.positions)

        def __getitem__(self, position):
            index = self.positions[position]
            return self.transform(Image.fromarray(images[index])), int(labels[index])

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(CachedImages(indices["train"], augment=True), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda", generator=generator)
    validation_loader = DataLoader(CachedImages(indices["val"]), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(CachedImages(indices["test"]), batch_size=batch_size, shuffle=False, num_workers=0)
    class_counts = np.bincount(labels[indices["train"]], minlength=2)
    weights = len(indices["train"]) / (2.0 * class_counts)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(network.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    def evaluate(loader):
        network.eval()
        probabilities, expected, total_loss, sample_count = [], [], 0.0, 0
        with torch.inference_mode():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = network(inputs)
                count = targets.size(0)
                total_loss += float(criterion(logits, targets).item()) * count
                sample_count += count
                probabilities.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
                expected.extend(targets.cpu().tolist())
        return np.asarray(probabilities), np.asarray(expected), total_loss / max(sample_count, 1)

    best_state, best_rank, best_epoch = None, None, 0
    curves = []
    started = time.perf_counter()
    finetuned = False
    for epoch in range(epochs):
        if epoch == 2:
            # Keep most ImageNet features fixed; adapt only the last two feature blocks at low LR.
            parameters = []
            for block in list(network.features.children())[-finetune_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
                    parameters.append(parameter)
            optimizer.add_param_group({"params": parameters, "lr": feature_lr})
            finetuned = True
        network.train()
        network.features.eval()
        if finetuned:
            for block in list(network.features.children())[-finetune_blocks:]:
                block.train()
        if freeze_batchnorm:
            for layer in network.features.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.eval()
        if selection == "balanced_pipeline" and epoch in (8, 14):
            for group in optimizer.param_groups:
                group["lr"] *= 0.3
        total_loss, samples_seen = 0.0, 0
        progress.update(f"MobileNetV3: epoch {epoch + 1}/{epochs}", model="mobilenet", epoch=epoch + 1, epochs=epochs, device=str(device))
        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = network(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            count = targets.size(0)
            total_loss += float(loss.item()) * count
            samples_seen += count
            if (batch_index + 1) % 25 == 0:
                progress.update(f"MobileNetV3: epoch {epoch + 1}/{epochs}, batch {batch_index + 1}/{len(train_loader)}", model="mobilenet", epoch=epoch + 1, epochs=epochs, batch=batch_index + 1, batches=len(train_loader), device=str(device))
        val_probabilities, val_labels, val_loss = evaluate(validation_loader)
        if selection == "balanced_pipeline":
            from .selection import choose_pipeline_threshold, selection_rank
            epoch_threshold, pipeline = choose_pipeline_threshold(val_labels, val_probabilities, UNCERTAINTY_THRESHOLD)
            rank = (*selection_rank(pipeline), -val_loss)
        else:
            epoch_threshold = 0.5
        val_metrics = _metrics(val_labels, val_probabilities, epoch_threshold)
        epoch_data = {
            "epoch": epoch + 1, "train_loss": total_loss / max(samples_seen, 1),
            "val_loss": val_loss, "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"], "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "threshold": epoch_threshold,
        }
        curves.append(epoch_data)
        if selection != "balanced_pipeline":
            rank = (val_metrics["f1"], val_metrics["balanced_accuracy"], -val_loss)
        else:
            epoch_data["val_pipeline"] = pipeline
        if best_rank is None or rank > best_rank:
            best_rank, best_epoch = rank, epoch + 1
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}
        progress.update(f"MobileNetV3: xong epoch {epoch + 1}/{epochs}, loss={epoch_data['train_loss']:.4f}, validation F1={val_metrics['f1']:.4f}", model="mobilenet", epoch=epoch + 1, epochs=epochs, validation_f1=val_metrics["f1"], device=str(device))
    training_seconds = time.perf_counter() - started
    network.load_state_dict(best_state)
    validation_probabilities, val_labels, _ = evaluate(validation_loader)
    if selection == "balanced_pipeline":
        threshold, validation_pipeline = choose_pipeline_threshold(val_labels, validation_probabilities, UNCERTAINTY_THRESHOLD)
        validation = _metrics(val_labels, validation_probabilities, threshold)
    else:
        threshold, validation = _choose_threshold(val_labels, validation_probabilities)
    # The held-out test loader is visited only after selecting epoch and threshold.
    test = {}
    if evaluate_test:
        test_probabilities, test_labels, _ = evaluate(test_loader)
        test = _metrics(test_labels, test_probabilities, threshold)
    artifact = "mobilenet.pt"
    torch.save({"state_dict": best_state, "class_names": list(CLASS_NAMES), "best_epoch": best_epoch, "image_size": IMAGE_SIZE}, run_dir / artifact)
    inference_ms = _cnn_latency(network, cnn_transform(), device, [records[index] for index in indices["val"]], torch)
    entry = {
        "name": "MobileNetV3 Small", "artifact": artifact, "test": test,
        "validation": validation, "decision_threshold": threshold,
        "uncertainty_threshold": UNCERTAINTY_THRESHOLD,
        "training_seconds": round(training_seconds, 4), "inference_ms": round(inference_ms, 4),
        "curves": curves, "best_epoch": best_epoch, "epochs": epochs,
        "batch_size": batch_size, "device": str(device), "pretrained": "ImageNet-1K",
        "class_weights": weights.tolist(), "finetune_after_epoch": 2 if epochs > 2 else None,
        "selection": selection, "finetune_blocks": finetune_blocks,
        "feature_lr": feature_lr, "freeze_batchnorm": freeze_batchnorm,
    }
    if selection == "balanced_pipeline":
        entry["validation_pipeline"] = validation_pipeline
    del network, optimizer, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    progress.update(f"Đã hoàn tất MobileNetV3; chọn epoch {best_epoch}, validation F1={validation['f1']:.4f}", model="mobilenet", epoch=epochs, epochs=epochs)
    return entry


def train_all(manifest_path=DEFAULT_MANIFEST, epochs=8, batch_size=8, device="auto", seed=42, artifact_root=None):
    if not 1 <= epochs <= 30:
        raise ValueError("epochs phải nằm trong khoảng 1..30.")
    if not 1 <= batch_size <= 32:
        raise ValueError("batch_size phải nằm trong khoảng 1..32.")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device cần là auto, cpu hoặc cuda.")
    artifact_root = Path(artifact_root or DEFAULT_ARTIFACT_ROOT).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = artifact_root / run_id
    run_dir.mkdir()
    progress = Progress(artifact_root, run_id)
    report = {"schema_version": 1, "status": "running", "run_id": run_id, "seed": seed, "models": {}}
    started = time.perf_counter()
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        import sklearn
        import cv2

        random.seed(seed)
        np.random.seed(seed)
        cv2.setNumThreads(2)
        progress.update("Đang kiểm tra manifest và phân chia dữ liệu", model="preparation")
        manifest, records, counts = load_manifest(manifest_path)
        report["dataset"] = {
            "image_count": len(records),
            **{split: sum(labels.values()) for split, labels in counts.items()},
            "counts": counts, "group_count": len({str(row["group"]) for row in records}),
            "source": manifest.get("source_url", manifest.get("source", "Kaggle")),
            "scope": manifest.get("scope", ""), "split_note": manifest.get("split_note", ""),
            "manifest": os.path.relpath(Path(manifest_path).resolve(), APP_DIR),
            "crop_preprocessing": manifest.get("crop_preprocessing", {}),
        }
        preparation_started = time.perf_counter()
        report["input_mode"] = manifest.get("input_mode", "full_image")
        features, images, labels, indices = _prepare(records, progress)
        report["preparation_seconds"] = round(time.perf_counter() - preparation_started, 4)
        svm = make_pipeline(StandardScaler(), SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed))
        forest = RandomForestClassifier(n_estimators=160, max_depth=12, class_weight="balanced", random_state=seed, n_jobs=2)
        for model_id, model in (("svm", svm), ("random_forest", forest)):
            report["models"][model_id] = _train_classical(model_id, model, features, labels, indices, records, run_dir, progress)
        report["models"]["mobilenet"] = _train_cnn(images, labels, indices, records, run_dir, progress, epochs, batch_size, device, seed)
        report.update({
            "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
            "total_seconds": round(time.perf_counter() - started, 4),
            "class_names": list(CLASS_NAMES), "feature_version": FEATURE_VERSION,
            "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
            "evaluation_note": "Epoch và ngưỡng chọn bằng validation. Precision/recall/F1 lấy defect làm lớp dương, trước bước từ chối điểm dưới 0.65. Test là khối khung hình cùng nguồn đã dùng trong phát triển, chưa phải đánh giá mù trên chai hoặc phiên chụp độc lập. Điểm mô hình chưa được hiệu chuẩn.",
            "latency_note": "Thời gian phân loại trên vùng chai, không gồm đọc tệp hoặc bộ định vị chai. CNN đo trên GPU huấn luyện, mô hình cổ điển trên CPU; web ghi thời gian toàn bộ yêu cầu thực tế trên CPU.",
        })
        _atomic_json(run_dir / "report.json", report)
        progress.update("Đã huấn luyện và đánh giá xong cả ba mô hình.", state="complete", report=str(run_dir / "report.json"), total_seconds=report["total_seconds"])
        # Make publishing the final operation: any earlier error leaves the old pointer intact.
        _atomic_json(artifact_root / "latest.json", {"run_id": run_id})
        return report
    except BaseException as error:
        report.update({"status": "failed", "error": str(error), "failed_at": datetime.now(timezone.utc).isoformat()})
        _atomic_json(run_dir / "report.json", report)
        progress.update(f"Huấn luyện dừng: {error}", state="failed", error=str(error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train HOG+SVM, HOG+RandomForest and MobileNetV3 using a grouped manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    try:
        report = train_all(args.manifest, args.epochs, args.batch_size, args.device, args.seed, args.artifact_root)
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps({"status": report["status"], "run_id": report["run_id"], "report": str(args.artifact_root / report["run_id"] / "report.json")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
