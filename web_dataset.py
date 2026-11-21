"""Read-only, indexed dataset metadata for the website; pixels never enter this cache."""
import json
import stat
import threading
from dataclasses import dataclass
from pathlib import Path

from ml.defect_types import TYPE_CODES, TYPE_NAMES

SPLITS = ("train", "val", "test")
LABEL_FILTERS = frozenset(("all", "defect", *TYPE_CODES))


class InvalidSampleFilter(ValueError):
    """A gallery filter is outside the supported source labels or splits."""


@dataclass(frozen=True)
class _Snapshot:
    manifest: dict
    splits: dict
    groups: dict
    records_by_id: dict


class DatasetCatalog:
    """Reload atomically when file identity, timestamp or size changes.

    Each app owns one catalog shared by status, gallery and training validation.
    Only metadata is cached; image files are still checked on every GET.
    """

    def __init__(self, manifest_path):
        self.path = Path(manifest_path)
        self._lock = threading.RLock()
        self._stamp = None
        self._cached = None

    def _signature(self):
        try:
            info = self.path.stat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(info.st_mode):
            return None
        return info.st_mtime_ns, info.st_ctime_ns, info.st_size, info.st_ino

    def _snapshot(self):
        with self._lock:
            stamp = self._signature()
            if self._cached is not None and stamp == self._stamp:
                return self._cached
            manifest = (json.loads(self.path.read_text(encoding="utf-8")) if stamp is not None
                        else {"records": [], "counts": {}})
            splits = {split: [] for split in SPLITS}
            by_id = {}
            for row in manifest.get("records", []):
                if row.get("split") in splits:
                    splits[row["split"]].append(row)
                if "id" in row:
                    by_id.setdefault(row["id"], row)
            groups = {split: {code: [] for code in TYPE_CODES} for split in SPLITS}
            known_types = set(TYPE_CODES[1:])
            for split, rows in splits.items():
                for row in rows:
                    if row["label"] == "normal":
                        groups[split]["normal"].append(row)
                    elif row["label"] == "defect":
                        for code in set(row.get("defect_types", [])) & known_types:
                            groups[split][code].append(row)
            # Publish only a fully parsed/indexed snapshot. If a file is replaced
            # while it is being read, its next stat will force another reload.
            self._cached = _Snapshot(manifest, splits, groups, by_id)
            self._stamp = stamp
            return self._cached

    @property
    def input_mode(self):
        return self._snapshot().manifest.get("input_mode")

    def summary(self, scope):
        snapshot = self._snapshot()
        manifest = snapshot.manifest
        counts = {split: len(rows) for split, rows in snapshot.splits.items()}
        return {"image_count": len(manifest.get("records", [])), **counts,
                "counts": manifest.get("counts", counts),
                "source": manifest.get("source_url", manifest.get("source", "Kaggle")),
                "scope": scope, "split_note": manifest.get("split_note", "")}

    def browse(self, split="test", label="all", limit=60, offset=0):
        if split not in SPLITS or label not in LABEL_FILTERS:
            raise InvalidSampleFilter("Bộ lọc dữ liệu không hợp lệ.")
        snapshot = self._snapshot()
        groups = snapshot.groups[split]
        if label == "all":
            records = snapshot.splits[split]
        elif label == "defect":
            records = [row for row in snapshot.splits[split] if row["label"] == "defect"]
        else:
            records = groups[label]
        categories = [{"code": code, "name": "Bình thường" if code == "normal" else TYPE_NAMES[code],
                       "count": len(rows), "preview_url": f"/api/samples/{rows[0]['id']}/image" if rows else None}
                      for code, rows in groups.items()]
        return {"total": len(records), "categories": categories, "samples": [
            {"id": row["id"], "image_url": f"/api/samples/{row['id']}/image", "label": row["label"],
             "split": row["split"], "filename": Path(row["path"]).name, "defect_types": row.get("defect_types", [])}
            for row in records[offset:offset + limit]]}

    def find(self, sample_id):
        """Return source metadata for the image route, which still validates its path."""
        return self._snapshot().records_by_id.get(sample_id)
