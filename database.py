import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "inspection.db"

PRODUCTS = [
    (2, "BOTTLE", "Chai nhựa", "Kiểm tra thiếu nắp, méo thân, bất thường bề mặt"),
]

DEFECTS = [
    (4, 2, "BOTTLE_CAP", "Thiếu / sai nắp chai"),
    (5, 2, "BOTTLE_SHAPE", "Thân chai méo / sai hình dạng"),
    (6, 2, "BOTTLE_SURFACE", "Bề mặt chai bất thường"),
]


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


@contextmanager
def _connection():
    con = connect()
    try:
        with con:
            yield con
    finally:
        con.close()


def init_db():
    with _connection() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS product_types(
            product_id INTEGER PRIMARY KEY,
            code TEXT UNIQUE NOT NULL,
            name TEXT UNIQUE NOT NULL,
            description TEXT
        );
        CREATE TABLE IF NOT EXISTS reference_images(
            reference_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            image_name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            FOREIGN KEY(product_id) REFERENCES product_types(product_id)
        );
        CREATE TABLE IF NOT EXISTS inspections(
            inspection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            reference_id INTEGER,
            image_name TEXT NOT NULL,
            image_path TEXT,
            result_path TEXT,
            status TEXT NOT NULL,
            metrics_json TEXT,
            checked_at TEXT NOT NULL,
            FOREIGN KEY(product_id) REFERENCES product_types(product_id),
            FOREIGN KEY(reference_id) REFERENCES reference_images(reference_id)
        );
        CREATE TABLE IF NOT EXISTS defect_types(
            defect_id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            FOREIGN KEY(product_id) REFERENCES product_types(product_id)
        );
        CREATE TABLE IF NOT EXISTS inspection_defects(
            inspection_id INTEGER NOT NULL,
            defect_id INTEGER NOT NULL,
            defect_score REAL,
            PRIMARY KEY(inspection_id, defect_id),
            FOREIGN KEY(inspection_id) REFERENCES inspections(inspection_id) ON DELETE CASCADE,
            FOREIGN KEY(defect_id) REFERENCES defect_types(defect_id)
        );
        """)
        con.executemany("INSERT OR IGNORE INTO product_types(product_id,code,name,description) VALUES(?,?,?,?)", PRODUCTS)
        con.executemany("INSERT OR IGNORE INTO defect_types(defect_id,product_id,code,name) VALUES(?,?,?,?)", DEFECTS)


def _product_id(con, product_name):
    row = con.execute(
        "SELECT product_id FROM product_types WHERE name=? AND code='BOTTLE'",
        (product_name,),
    ).fetchone()
    if not row:
        raise ValueError(f"Loại sản phẩm chưa được khai báo: {product_name}")
    return row[0]


def ensure_reference(product_name, image_path):
    if not image_path:
        return None
    image_path = str(image_path)
    image_name = Path(image_path).name
    with _connection() as con:
        pid = _product_id(con, product_name)
        row = con.execute("SELECT reference_id FROM reference_images WHERE product_id=? AND image_path=?", (pid, image_path)).fetchone()
        if row:
            return row[0]
        cur = con.execute(
            "INSERT INTO reference_images(product_id,image_name,image_path,is_active,created_at) VALUES(?,?,?,?,?)",
            (pid, image_name, image_path, 1, datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        )
        return cur.lastrowid


def save_inspection(product_name, reference_path, image_name, image_path, status, metrics, defects, defect_scores, result_path):
    with _connection() as con:
        pid = _product_id(con, product_name)
        ref_id = None
        if reference_path:
            ref_name = Path(reference_path).name
            row = con.execute("SELECT reference_id FROM reference_images WHERE product_id=? AND image_path=?", (pid, str(reference_path))).fetchone()
            if row:
                ref_id = row[0]
            else:
                cur = con.execute(
                    "INSERT INTO reference_images(product_id,image_name,image_path,is_active,created_at) VALUES(?,?,?,?,?)",
                    (pid, ref_name, str(reference_path), 1, datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                )
                ref_id = cur.lastrowid

        cur = con.execute("""
            INSERT INTO inspections(product_id,reference_id,image_name,image_path,result_path,status,metrics_json,checked_at)
            VALUES(?,?,?,?,?,?,?,?)
        """, (
            pid, ref_id, image_name, str(image_path or "camera"), str(result_path), status,
            json.dumps(metrics, ensure_ascii=False), datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        ))
        inspection_id = cur.lastrowid
        for defect in defects:
            d = con.execute("SELECT defect_id FROM defect_types WHERE product_id=? AND name=?", (pid, defect)).fetchone()
            if d:
                con.execute("INSERT OR REPLACE INTO inspection_defects(inspection_id,defect_id,defect_score) VALUES(?,?,?)",
                            (inspection_id, d[0], float(defect_scores.get(defect, 0.0))))
        return inspection_id


def get_history(limit=300):
    with _connection() as con:
        return con.execute("""
        SELECT i.inspection_id AS id, p.name AS product_name, i.image_name, i.status,
               i.metrics_json, i.checked_at, i.result_path,
               COALESCE(GROUP_CONCAT(d.name, '; '), 'Không có') AS defects
        FROM inspections i
        JOIN product_types p ON p.product_id=i.product_id
        LEFT JOIN inspection_defects x ON x.inspection_id=i.inspection_id
        LEFT JOIN defect_types d ON d.defect_id=x.defect_id
        WHERE p.code='BOTTLE'
        GROUP BY i.inspection_id
        ORDER BY i.inspection_id DESC
        LIMIT ?
        """, (limit,)).fetchall()


def delete_inspection(row_id):
    with _connection() as con:
        con.execute("""
            DELETE FROM inspections WHERE inspection_id=?
            AND product_id IN (SELECT product_id FROM product_types WHERE code='BOTTLE')
        """, (row_id,))


def delete_all_inspections():
    """Clear bottle history while preserving configuration and older product records."""
    with _connection() as con:
        con.execute("""
            DELETE FROM inspection_defects WHERE inspection_id IN (
                SELECT inspection_id FROM inspections WHERE product_id IN (
                    SELECT product_id FROM product_types WHERE code='BOTTLE'
                )
            )
        """)
        con.execute("""
            DELETE FROM inspections WHERE product_id IN (
                SELECT product_id FROM product_types WHERE code='BOTTLE'
            )
        """)
