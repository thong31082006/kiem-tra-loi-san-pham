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

    def connect(self):
        return sqlite3.connect(self.db_path, timeout=20)

    def save(self, item):
        with closing(self.connect()) as con, con:
            con.execute("INSERT INTO predictions VALUES(?,?,?,?)", (
                item["id"], item["created_at"], item["filename"], json.dumps(item, ensure_ascii=False)
            ))

    def list(self, limit=100):
        with closing(self.connect()) as con:
            rows = con.execute("SELECT payload FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def get(self, item_id):
        with closing(self.connect()) as con:
            row = con.execute("SELECT payload FROM predictions WHERE id=?", (item_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def image_path(self, item_id):
        return self.upload_dir / f"{item_id}.jpg"

    def delete(self, item_id):
        item = self.get(item_id)
        if item:
            with closing(self.connect()) as con, con:
                con.execute("DELETE FROM predictions WHERE id=?", (item_id,))
            self.image_path(item_id).unlink(missing_ok=True)
        return item is not None
