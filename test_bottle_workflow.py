"""Bottle regression checks; all saved images/history use temporary directories.

Run ``python test_bottle_workflow.py`` for detector/database checks, or append
``--gui`` to also exercise the Tkinter workflow on a machine with a display.
"""
import json
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np

import database
import result
from image_input import load_image
from products import PRODUCTS, bottle_detector

APP_DIR = Path(__file__).resolve().parent
REFERENCE = APP_DIR / "reference" / "chai_chuan.png"
GOOD = APP_DIR / "images" / "demo" / "chai_dat.png"
BAD = APP_DIR / "images" / "demo" / "chai_loi.png"
GUI_ENABLED = "--gui" in sys.argv
if GUI_ENABLED:
    sys.argv.remove("--gui")


class BottleWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.folder.name)
        self.patches = [
            patch.object(database, "DB_PATH", self.temp_path / "inspection.db"),
            patch.object(result, "__file__", str(self.temp_path / "result.py")),
        ]
        for item in self.patches:
            item.start()
        database.init_db()
        self.reference = load_image(str(REFERENCE))

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.folder.cleanup()

    def rows(self, sql):
        with closing(database.connect()) as con:
            return [tuple(row) for row in con.execute(sql)]

    def record_callback_errors(self, app):
        errors = []
        app.report_callback_exception = lambda error_type, error, traceback: errors.append(f"{error_type.__name__}: {error}")
        self.addCleanup(self.assertEqual, errors, [], "Unexpected error in a Tk event/button callback")

    def test_only_bottle_is_registered_and_seeded(self):
        self.assertEqual(list(PRODUCTS), [bottle_detector.PRODUCT_NAME])
        self.assertEqual(self.rows("SELECT product_id, code FROM product_types"), [(2, "BOTTLE")])
        self.assertEqual(self.rows("SELECT defect_id, product_id FROM defect_types ORDER BY defect_id"), [(4, 2), (5, 2), (6, 2)])
        database.init_db()
        self.assertEqual(len(self.rows("SELECT * FROM product_types")), 1)

    def test_demo_files_pass_and_fail(self):
        normal = bottle_detector.inspect(self.reference, load_image(str(GOOD)))
        self.assertEqual(normal.status, "ĐẠT")
        self.assertEqual(normal.defects, [])
        defective = bottle_detector.inspect(self.reference, load_image(str(BAD)))
        self.assertEqual(defective.status, "KHÔNG ĐẠT")
        self.assertEqual(set(defective.defects), {row[3] for row in database.DEFECTS})
        for visual in (defective.result_image, defective.mask_image, defective.edge_image, defective.diff_image):
            self.assertIsNotNone(visual)
            self.assertEqual(visual.ndim, 3)

    def test_empty_background_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Không tách được sản phẩm"):
            bottle_detector.inspect(self.reference, np.full_like(self.reference, 255))

    def test_save_unicode_image_and_metrics(self):
        image_path = self.temp_path / "ảnh chai nhựa.png"
        result.imwrite_unicode(image_path, self.reference)
        checked = bottle_detector.inspect(self.reference, load_image(str(image_path)))
        saved = result.save_result("Chai nhựa", str(REFERENCE), str(image_path), checked)
        self.assertTrue(Path(saved).is_file())
        self.assertGreater(load_image(saved).size, 0)
        row = database.get_history()[0]
        self.assertEqual(row["product_name"], "Chai nhựa")
        self.assertEqual(row["image_name"], image_path.name)
        self.assertEqual(row["status"], "ĐẠT")
        self.assertEqual(json.loads(row["metrics_json"]), checked.metrics)

    def test_legacy_history_is_preserved_and_hidden(self):
        with closing(database.connect()) as con, con:
            con.execute("INSERT INTO product_types VALUES(99, 'LEGACY', 'Sản phẩm cũ', 'Archived')")
            con.execute("INSERT INTO defect_types VALUES(99, 99, 'LEGACY_DEFECT', 'Lỗi cũ', NULL)")
            cursor = con.execute("""
                INSERT INTO inspections(product_id, image_name, status, checked_at)
                VALUES(99, 'old.png', 'KHÔNG ĐẠT', '01/01/2026 00:00:00')
            """)
            old_id = cursor.lastrowid
            con.execute("INSERT INTO inspection_defects VALUES(?, 99, 12)", (old_id,))
        database.init_db()
        self.assertEqual(database.get_history(), [])
        database.delete_inspection(old_id)
        self.assertEqual(self.rows("SELECT inspection_id FROM inspections"), [(old_id,)])
        with self.assertRaises(ValueError):
            database.ensure_reference("Sản phẩm cũ", "old.png")

        defective = bottle_detector.inspect(self.reference, load_image(str(BAD)))
        result.save_result("Chai nhựa", str(REFERENCE), str(BAD), defective)
        current = database.get_history()
        self.assertEqual(len(current), 1)
        configuration = {table: self.rows("SELECT * FROM " + table) for table in ("product_types", "defect_types", "reference_images")}
        database.delete_inspection(current[0]["id"])
        self.assertEqual(database.get_history(), [])
        self.assertEqual(self.rows("SELECT inspection_id, defect_id FROM inspection_defects"), [(old_id, 99)])

        result.save_result("Chai nhựa", str(REFERENCE), str(BAD), defective)
        database.delete_all_inspections()
        self.assertEqual(database.get_history(), [])
        self.assertEqual(self.rows("SELECT inspection_id FROM inspections"), [(old_id,)])
        self.assertEqual(self.rows("SELECT inspection_id, defect_id FROM inspection_defects"), [(old_id, 99)])
        for table, expected in configuration.items():
            self.assertEqual(self.rows("SELECT * FROM " + table), expected)

    @unittest.skipUnless(GUI_ENABLED, "Append --gui to exercise the desktop interface")
    def test_desktop_inspect_replace_save_and_history(self):
        import desktop_legacy as main

        def fail_dialog(title, message, **kwargs):
            raise AssertionError(f"{title}: {message}")

        def assert_result_cleared(app):
            self.assertIsNone(app.inspection)
            self.assertFalse(app.metrics_tree.get_children())
            self.assertTrue(app.save_button.instate(("disabled",)))
            for name in (*main.PROCESS_VIEWS, "Kết quả"):
                self.assertIsNone(app.views[name].image, name)

        with patch.object(main.messagebox, "showerror", side_effect=fail_dialog), patch.object(main.messagebox, "showwarning", side_effect=fail_dialog):
            app = main.App()
            self.record_callback_errors(app)
            try:
                app.update()
                self.assertEqual(app.product_name.get(), "Chai nhựa")
                with patch.object(app, "_pick_image", return_value=str(REFERENCE)):
                    app.choose_reference()
                with patch.object(app, "_pick_image", return_value=str(BAD)):
                    app.choose_test()
                app.inspect()
                app.update()
                self.assertEqual(app.inspection.status, "KHÔNG ĐẠT")
                self.assertEqual(len(app.metrics_tree.get_children()), 3)
                self.assertFalse(app.save_button.instate(("disabled",)))
                for name in (*main.PROCESS_VIEWS, "Kết quả"):
                    self.assertIsNotNone(app.views[name].image, name)

                with patch.object(app, "_pick_image", return_value=str(REFERENCE)):
                    app.choose_reference()
                assert_result_cleared(app)
                app.inspect()
                self.assertEqual(app.inspection.status, "KHÔNG ĐẠT")

                with patch.object(app, "_pick_image", return_value=str(GOOD)):
                    app.choose_test()
                assert_result_cleared(app)
                app.inspect()
                self.assertEqual(app.inspection.status, "ĐẠT")
                with patch.object(main.messagebox, "showinfo"):
                    app.save_current()
                self.assertEqual(database.get_history()[0]["status"], "ĐẠT")

                with patch.dict(app.capture_camera.__globals__, capture_from_camera=lambda *args: self.reference.copy()), patch.object(main.messagebox, "showinfo"):
                    app.capture_camera()
                self.assertEqual(app.test_path, "camera")
                assert_result_cleared(app)
                app.inspect()
                self.assertEqual(app.inspection.status, "ĐẠT")

                app.open_history()
                app.update()

                def descendants(widget):
                    for child in widget.winfo_children():
                        yield child
                        yield from descendants(child)

                history = next(widget for widget in app.winfo_children() if isinstance(widget, main.tk.Toplevel))
                widgets = list(descendants(history))
                tree = next(widget for widget in widgets if isinstance(widget, main.ttk.Treeview))
                self.assertEqual(len(tree.get_children()), 1)
                buttons = {widget.cget("text"): widget for widget in widgets if isinstance(widget, (main.tk.Button, main.ttk.Button))}
                clear_button = buttons["Xóa lịch sử chai nhựa"]
                with patch.object(main.messagebox, "askyesno", return_value=False):
                    clear_button.invoke()
                self.assertEqual(len(database.get_history()), 1)
                with patch.object(main.messagebox, "askyesno", return_value=True), patch.object(main.messagebox, "showinfo"):
                    clear_button.invoke()
                self.assertEqual(database.get_history(), [])
                app.clear_test()
                self.assertIsNone(app.test)
                assert_result_cleared(app)
            finally:
                app.destroy()

    @unittest.skipUnless(GUI_ENABLED, "Append --gui to exercise the desktop interface")
    def test_kaggle_library_filter_preview_and_input_selection(self):
        import desktop_legacy as main
        from bottle_dataset import list_samples

        samples = list_samples()
        if not samples:
            self.skipTest("Download the Kaggle images before testing the image library")

        def fail_dialog(title, message, **kwargs):
            raise AssertionError(f"{title}: {message}")

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        with patch.object(main.messagebox, "showerror", side_effect=fail_dialog):
            app = main.App()
            self.record_callback_errors(app)
            try:
                app.update()
                app.inspect()
                self.assertIsNotNone(app.inspection)
                app.open_samples()
                app.update()
                library = next(widget for widget in app.winfo_children() if isinstance(widget, main.tk.Toplevel))
                widgets = list(descendants(library))
                tree = next(widget for widget in widgets if isinstance(widget, main.ttk.Treeview))
                filters = next(widget for widget in widgets if isinstance(widget, main.ttk.Combobox))
                preview = next(widget for widget in widgets if isinstance(widget, main.ImagePreview))
                self.assertEqual(len(tree.get_children()), len(samples))
                self.assertIsNotNone(preview.image)
                labels = {sample["label"] for sample in samples}
                self.assertEqual(set(filters.cget("values")), labels | {"Tất cả"})
                for label in labels:
                    filters.set(label)
                    filters.event_generate("<<ComboboxSelected>>")
                    app.update()
                    self.assertEqual(len(tree.get_children()), sum(sample["label"] == label for sample in samples))
                    self.assertTrue(all(tree.item(item, "values")[1] == label for item in tree.get_children()))

                selected = samples[int(tree.selection()[0])]
                self.assertIsNotNone(preview.image)
                buttons = {widget.cget("text"): widget for widget in widgets if isinstance(widget, main.ttk.Button)}
                buttons["Dùng làm ảnh kiểm tra"].invoke()
                self.assertEqual(app.test_path, selected["path"])
                self.assertIsNone(app.inspection)
                self.assertTrue(app.save_button.instate(("disabled",)))
                for name in (*main.PROCESS_VIEWS, "Kết quả"):
                    self.assertIsNone(app.views[name].image, name)

                app.load_demo(use_dataset=False)
                app.inspect()
                self.assertIsNotNone(app.inspection)
                buttons["Dùng làm ảnh chuẩn"].invoke()
                self.assertEqual(app.reference_path, selected["path"])
                self.assertIsNone(app.inspection)
                self.assertFalse(app.metrics_tree.get_children())
                self.assertTrue(app.save_button.instate(("disabled",)))
            finally:
                app.destroy()

    @unittest.skipUnless(GUI_ENABLED, "Append --gui to exercise the desktop interface")
    def test_main_layout_at_default_and_minimum_window_sizes(self):
        import desktop_legacy as main

        app = main.App()
        self.record_callback_errors(app)
        try:
            app.update()
            self.assertEqual((app.winfo_width(), app.winfo_height()), (1320, 820))
            sidebar = app.inspect_button.master
            buttons = [widget for widget in sidebar.winfo_children() if isinstance(widget, main.ttk.Button)]
            self.assertTrue(buttons)
            for width, height in ((1320, 820), (1080, 700)):
                with self.subTest(width=width, height=height):
                    app.geometry(f"{width}x{height}")
                    app.update()
                    self.assertEqual((app.winfo_width(), app.winfo_height()), (width, height))
                    for button in buttons:
                        self.assertTrue(button.winfo_ismapped(), button.cget("text"))
                        self.assertGreater(button.winfo_height(), 1)
                        self.assertGreaterEqual(button.winfo_x(), 0)
                        self.assertGreaterEqual(button.winfo_y(), 0)
                        self.assertLessEqual(button.winfo_x() + button.winfo_width(), sidebar.winfo_width(), button.cget("text"))
                        self.assertLessEqual(button.winfo_y() + button.winfo_height(), sidebar.winfo_height(), button.cget("text"))
                    self.assertTrue(app.footer.winfo_ismapped())
                    self.assertGreater(app.footer.winfo_height(), 1)
                    self.assertGreaterEqual(app.footer.winfo_y(), 0)
                    self.assertLessEqual(app.footer.winfo_y() + app.footer.winfo_height(), height)
                    for index, names in ((0, ("Ảnh chuẩn", "Ảnh kiểm tra", "Kết quả")), (1, main.PROCESS_VIEWS)):
                        app.notebook.select(index)
                        app.update()
                        for name in names:
                            preview = app.views[name]
                            self.assertTrue(preview.winfo_ismapped(), name)
                            self.assertGreater(preview.canvas.winfo_width(), 50, name)
                            self.assertGreater(preview.canvas.winfo_height(), 40, name)
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)
