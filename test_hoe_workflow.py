"""Regression checks using a temporary database; never touches real history."""
import gc
import json
import runpy
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import main
import database
import result
from image_input import load_image
from products import hoe_detector as hoe


def run_checks():
    with tempfile.TemporaryDirectory() as folder:
        with patch.object(database, 'DB_PATH', Path(folder) / 'inspection.db'), patch.object(result, '__file__', str(Path(folder) / 'result.py')):
            product = hoe.PRODUCT_NAME
            passed = '\u0110\u1ea0T'
            ref_path, test_path = main.DEMO[product]
            ref = load_image(str(ref_path))
            same = hoe.inspect(ref, ref)
            assert same.status == passed, same.metrics
            h, w = ref.shape[:2]
            for scale, angle, dx, dy in [(0.65, 0, 35, -20), (0.8, 8, 10, 10), (0.8, -8, -10, 10)]:
                matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
                matrix[:, 2] += [dx, dy]
                transformed = cv2.warpAffine(ref, matrix, (w, h), borderValue=tuple(int(v) for v in ref[0, 0]))
                checked = hoe.inspect(ref, transformed)
                shape = next(iter(checked.metrics.values()))
                print('Transform:', scale, angle, 'shape:', shape)
                assert shape >= hoe.HOE_MIN_SHAPE_SIMILARITY
            for shape, rust, scratch in [(95,.4,.8), (87,0,.5), (96,8,.7), (96,.2,4.3), (82,12,6), (90,1,2)]:
                mask = np.full((100,100), 255, np.uint8)
                blank = np.zeros((100,100,3), np.uint8)
                rustmask = np.zeros_like(mask)
                rustmask.flat[:round(rust*100)] = 255
                scratchmask = np.zeros_like(mask)
                scratchmask.flat[:round(scratch*100)] = 255
                with patch.object(hoe, '_align_hoe', return_value=(blank,blank,mask,mask,shape,np.zeros_like(mask))), patch.object(hoe, '_rust_mask', return_value=rustmask), patch.object(hoe, '_scratch_mask', side_effect=[(np.zeros_like(mask),mask),(scratchmask,mask)]):
                    checked = hoe.inspect(blank, blank)
                count = int(shape<90) + int(rust>1) + int(scratch>2)
                assert len(checked.defects) == count
                assert (checked.status == passed) == (count == 0)
            database.init_db()
            saved = result.save_result(product, ref_path, 'camera', same)
            assert Path(saved).exists()
            row = database.get_history()[0]
            assert row['status'] == same.status
            assert json.loads(row['metrics_json']) == same.metrics
            bad = hoe.inspect(ref, load_image(str(test_path)))
            result.save_result(product, ref_path, 'camera', bad)
            con = database.connect()
            config = {t: [tuple(r) for r in con.execute('SELECT * FROM '+t)] for t in ('product_types','defect_types','reference_images')}
            con.close()

            def exercise(app):
                app.update()
                app.inspect()
                app.update()
                assert app.inspection is not None
                assert app.result_details.cget('text')
                app.open_history()
                app.update()

                def descendants(widget):
                    for child in widget.winfo_children():
                        yield child
                        yield from descendants(child)

                win = next(w for w in app.winfo_children() if isinstance(w, main.tk.Toplevel))
                widgets = list(descendants(win))
                tree = next(w for w in widgets if isinstance(w, main.ttk.Treeview))
                buttons = [w for w in widgets if isinstance(w, main.tk.Button)]
                tree.selection_set(tree.get_children()[0])
                with patch.object(main.messagebox, 'askyesno', return_value=True):
                    buttons[1].invoke()
                assert len(database.get_history()) == 1
                with patch.object(main.messagebox, 'askyesno', return_value=False):
                    buttons[3].invoke()
                assert len(database.get_history()) == 1
                with patch.object(main.messagebox, 'askyesno', return_value=True), patch.object(main.messagebox, 'showinfo') as info:
                    buttons[3].invoke()
                    info.assert_called_once()
                assert not tree.get_children() and not database.get_history()
                database.delete_all_inspections()
                con = database.connect()
                assert con.execute('SELECT COUNT(*) FROM inspection_defects').fetchone()[0] == 0
                for table, rows in config.items():
                    assert [tuple(r) for r in con.execute('SELECT * FROM '+table)] == rows
                con.close()
                with patch.object(app, '_pick_image', return_value=str(ref_path)):
                    app.choose_reference()
                    app.choose_test()
                assert app.reference is not None and app.test is not None
                with patch.dict(app.capture_camera.__globals__, capture_from_camera=lambda *args: ref.copy()), patch.object(main.messagebox, 'showinfo'):
                    app.capture_camera()
                assert app.test_path == 'camera'
                app.inspect()
                with patch.object(main.messagebox, 'showinfo'):
                    app.save_current()
                assert database.get_history()[0]['status'] == passed
                app.clear_test()
                assert app.test is None
                app.destroy()

            def fail_dialog(title, message, **kwargs):
                raise AssertionError(str(message))

            with patch.object(main.tk.Tk, 'mainloop', exercise), patch.object(main.messagebox, 'showerror', side_effect=fail_dialog), patch.object(main.messagebox, 'showwarning', side_effect=fail_dialog):
                runpy.run_path(str(main.APP_DIR / 'main.py'), run_name='__main__')
            gc.collect()
    print('PASS: imports, normalization, threshold cases, save, Tk result, delete row, delete all Yes/No, preserved configuration; main.py executed.')


if __name__ == '__main__':
    run_checks()
