"""History for trained, single-image predictions, separate from the old comparison app."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent


class HistoryStore:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir or APP_DIR / "data" / "web").resolve()
        self.upload_dir = self.data_dir / "images"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "history.db"
        with closing(self.connect()) as con, con:
            con.execute("""CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                filename TEXT NOT NULL, payload TEXT NOT NULL
            )""")
            con.execute("""CREATE TABLE IF NOT EXISTS reviews (
                prediction_id TEXT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
                target TEXT NOT NULL, verdict TEXT NOT NULL,
                corrected_label TEXT, corrected_type TEXT, note TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                PRIMARY KEY(prediction_id, target)
            )""")
            con.execute("""CREATE TABLE IF NOT EXISTS batch_requests (
                batch_id TEXT NOT NULL, position INTEGER NOT NULL,
                prediction_id TEXT NOT NULL UNIQUE REFERENCES predictions(id) ON DELETE CASCADE,
                content_hash TEXT NOT NULL, model TEXT NOT NULL,
                PRIMARY KEY(batch_id, position)
            )""")

    def connect(self):
        con = sqlite3.connect(self.db_path, timeout=20)
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def save(self, item, batch=None):
        with closing(self.connect()) as con, con:
            con.execute("INSERT INTO predictions VALUES(?,?,?,?)", (
                item["id"], item["created_at"], item["filename"], json.dumps(item, ensure_ascii=False)
            ))
            if batch:
                con.execute("INSERT INTO batch_requests VALUES(?,?,?,?,?)", (
                    batch["batch_id"], batch["position"], item["id"], batch["content_hash"], item["model"]
                ))

    def list(self, limit=100, review="all"):
        with closing(self.connect()) as con:
            condition, arguments = "", []
            if review == "unreviewed":
                condition = " WHERE NOT EXISTS (SELECT 1 FROM reviews r WHERE r.prediction_id=p.id)"
            elif review != "all":
                condition = " WHERE EXISTS (SELECT 1 FROM reviews r WHERE r.prediction_id=p.id AND r.verdict=?)"
                arguments.append(review)
            rows = con.execute("SELECT payload FROM predictions p" + condition + " ORDER BY created_at DESC LIMIT ?", (*arguments, limit)).fetchall()
            return [self._with_reviews(con, json.loads(row[0])) for row in rows]

    @staticmethod
    def _with_reviews(con, item):
        rows = con.execute("SELECT target,verdict,corrected_label,corrected_type,note,updated_at FROM reviews WHERE prediction_id=?", (item["id"],)).fetchall()
        if rows:
            item["reviews"] = {row[0]: dict(zip(("target", "verdict", "corrected_label", "corrected_type", "note", "updated_at"), row)) for row in rows}
        return item

    def get(self, item_id):
        with closing(self.connect()) as con:
            row = con.execute("SELECT payload FROM predictions WHERE id=?", (item_id,)).fetchone()
            return self._with_reviews(con, json.loads(row[0])) if row else None

    def save_review(self, item_id, review):
        with closing(self.connect()) as con, con:
            if not con.execute("SELECT 1 FROM predictions WHERE id=?", (item_id,)).fetchone():
                raise KeyError(item_id)
            con.execute("""INSERT INTO reviews VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(prediction_id,target) DO UPDATE SET
                verdict=excluded.verdict, corrected_label=excluded.corrected_label,
                corrected_type=excluded.corrected_type, note=excluded.note, updated_at=excluded.updated_at""",
                (item_id, review["target"], review["verdict"], review.get("corrected_label"), review.get("corrected_type"), review["note"], review["updated_at"]))
        return review

    def batch_result(self, batch_id, position):
        with closing(self.connect()) as con:
            row = con.execute("SELECT prediction_id,content_hash,model FROM batch_requests WHERE batch_id=? AND position=?", (batch_id, position)).fetchone()
        if not row:
            return None
        return {"item": self.get(row[0]), "content_hash": row[1], "model": row[2]}

    def image_path(self, item_id):
        return self.upload_dir / f"{item_id}.jpg"

    def delete(self, item_id):
        item = self.get(item_id)
        if item:
            with closing(self.connect()) as con, con:
                con.execute("DELETE FROM predictions WHERE id=?", (item_id,))
            self.image_path(item_id).unlink(missing_ok=True)
        return item is not None
