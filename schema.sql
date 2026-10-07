PRAGMA foreign_keys = ON;
CREATE TABLE product_types(
  product_id INTEGER PRIMARY KEY,
  code TEXT UNIQUE NOT NULL,
  name TEXT UNIQUE NOT NULL,
  description TEXT
);
CREATE TABLE reference_images(
  reference_id INTEGER PRIMARY KEY AUTOINCREMENT,
  product_id INTEGER NOT NULL,
  image_name TEXT NOT NULL,
  image_path TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  FOREIGN KEY(product_id) REFERENCES product_types(product_id)
);
CREATE TABLE inspections(
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
CREATE TABLE defect_types(
  defect_id INTEGER PRIMARY KEY,
  product_id INTEGER NOT NULL,
  code TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  FOREIGN KEY(product_id) REFERENCES product_types(product_id)
);
CREATE TABLE inspection_defects(
  inspection_id INTEGER NOT NULL,
  defect_id INTEGER NOT NULL,
  defect_score REAL,
  PRIMARY KEY(inspection_id, defect_id),
  FOREIGN KEY(inspection_id) REFERENCES inspections(inspection_id) ON DELETE CASCADE,
  FOREIGN KEY(defect_id) REFERENCES defect_types(defect_id)
);
