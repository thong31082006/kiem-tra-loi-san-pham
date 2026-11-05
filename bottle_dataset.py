"""Local Kaggle PET images and their supplied component annotations.

Component counts describe source annotations, not predictions by this application.
"""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATASET_DIR = APP_DIR / "data" / "datasets" / "plastic_bottles"
SOURCE_URL = "https://www.kaggle.com/datasets/dhiaalhemdani/industrial-inspection-system"
SOURCE_NAME = "Kaggle · Industrial Inspection System"
COMPONENT_NAMES = {0: "bottle", 1: "cap", 2: "label", 3: "liquid"}


def _component_counts(label_path):
    counts = {name: 0 for name in COMPONENT_NAMES.values()}
    if not label_path.is_file():
        return counts
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        if not values:
            continue
        try:
            component = COMPONENT_NAMES.get(int(values[0]))
        except ValueError:
            continue
        if component:
            counts[component] += 1
    return counts


def _label_for(counts):
    bottles = counts["bottle"]
    if not bottles:
        return "Chưa có nhãn thành phần"
    notes = []
    notes.append("Có nắp" if counts["cap"] >= bottles else "Thiếu nắp")
    notes.append("có nhãn" if counts["label"] >= bottles else "thiếu nhãn")
    if bottles > 1:
        notes.append(f"{bottles} chai")
    return ", ".join(notes)


def list_samples():
    """Return every downloaded image, with labels derived from YOLO annotations."""
    samples = []
    images = DATASET_DIR / "images"
    if not images.is_dir():
        return samples
    for path in sorted(images.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        relative = path.relative_to(images)
        counts = _component_counts((DATASET_DIR / "labels" / relative).with_suffix(".txt"))
        samples.append({
            "path": str(path.resolve()),
            "label": _label_for(counts),
            "source": SOURCE_NAME,
            "split": relative.parts[0],
            "components": counts,
        })
    return samples


def get_default_pair():
    """No reference/test pairing is supplied by this object-detection dataset.

    Users should select images of the same bottle, viewpoint, and lighting for
    reference comparison. The app's controlled demo remains the default.
    """
    return None
