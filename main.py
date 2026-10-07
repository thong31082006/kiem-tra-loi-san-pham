import base64
import json
import os
import platform
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2

from image_input import load_image, resize_for_processing, capture_from_camera
from image_processing import to_gray, make_visual_gray
from products import PRODUCTS
from database import init_db, get_history, delete_inspection, delete_all_inspections
from result import save_result

APP_DIR = Path(__file__).resolve().parent
BG = "#eef3f8"
NAVY = "#174f8f"
BLUE = "#1f6feb"
GREEN = "#159447"
RED = "#d62828"
DARK = "#1f2937"

DEMO = {
    "Cuốc sắt": (APP_DIR / "reference" / "cuoc_chuan.png", APP_DIR / "images" / "demo" / "cuoc_loi.png"),
    "Chai nhựa": (APP_DIR / "reference" / "chai_chuan.png", APP_DIR / "images" / "demo" / "chai_loi.png"),
    "Gạch men": (APP_DIR / "reference" / "gach_chuan.png", APP_DIR / "images" / "demo" / "gach_loi.png"),
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kiểm tra lỗi 3 sản phẩm - Nhóm 8")
        self.geometry("1460x900")
        self.minsize(1220, 760)
        self.configure(bg=BG)
        init_db()

        self.product_name = tk.StringVar(value="Cuốc sắt")
        self.reference_path = None
        self.test_path = None
        self.reference = None
        self.test = None
        self.inspection = None
        self._photos = []

        self._build_ui()
        self.load_demo()

    def _build_ui(self):
        header = tk.Frame(self, bg=NAVY, height=88)
        header.pack(fill="x")
        tk.Label(header, text="CHƯƠNG TRÌNH KIỂM TRA LỖI SẢN PHẨM", fg="white", bg=NAVY,
                 font=("Arial", 22, "bold")).pack(anchor="w", padx=25, pady=(13,0))
        tk.Label(header, text="Cuốc sắt • Chai nhựa • Gạch men | Python + Tkinter + OpenCV + NumPy + SQLite",
                 fg="#dfefff", bg=NAVY, font=("Arial", 10)).pack(anchor="w", padx=28, pady=(3,10))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)
        self.sidebar = tk.Frame(body, width=270, bg="white", bd=1, relief="solid")
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        main = tk.Frame(body, bg=BG)
        main.pack(side="left", fill="both", expand=True, padx=(0,10), pady=10)
        self._build_views(main)
        self._build_result_panel(main)

    def _build_sidebar(self):
        tk.Label(self.sidebar, text="1. LOẠI SẢN PHẨM", bg="white", fg=DARK, font=("Arial",11,"bold")).pack(anchor="w", padx=14, pady=(16,6))
        combo = ttk.Combobox(self.sidebar, textvariable=self.product_name, values=list(PRODUCTS.keys()), state="readonly")
        combo.pack(fill="x", padx=14, pady=(0,12))
        combo.bind("<<ComboboxSelected>>", lambda e: self.load_demo())

        ttk.Separator(self.sidebar).pack(fill="x", padx=14, pady=2)
        tk.Label(self.sidebar, text="2. ẢNH SẢN PHẨM", bg="white", fg=DARK, font=("Arial",11,"bold")).pack(anchor="w", padx=14, pady=(10,6))
        self.ref_label = tk.Label(self.sidebar, text="Ảnh chuẩn: chưa chọn", bg="white", fg="#555", wraplength=235, justify="left")
        self.ref_label.pack(anchor="w", padx=14)
        tk.Button(self.sidebar, text="Chọn ảnh CHUẨN", command=self.choose_reference, bg="#e8f1ff", height=2).pack(fill="x", padx=14, pady=(5,10))
        self.test_label = tk.Label(self.sidebar, text="Ảnh kiểm tra: chưa chọn", bg="white", fg="#555", wraplength=235, justify="left")
        self.test_label.pack(anchor="w", padx=14)
        tk.Button(self.sidebar, text="Chọn ảnh CẦN KIỂM TRA", command=self.choose_test, bg="#e8f1ff", height=2).pack(fill="x", padx=14, pady=(5,4))
        tk.Button(self.sidebar, text="Chụp từ CAMERA", command=self.capture_camera, bg="#ecfdf3", height=2).pack(fill="x", padx=14, pady=(0,10))
        tk.Button(self.sidebar, text="Nạp lại ảnh DEMO", command=self.load_demo, height=2).pack(fill="x", padx=14, pady=(0,12))

        ttk.Separator(self.sidebar).pack(fill="x", padx=14, pady=2)
        tk.Label(self.sidebar, text="3. THAO TÁC", bg="white", fg=DARK, font=("Arial",11,"bold")).pack(anchor="w", padx=14, pady=(10,6))
        tk.Button(self.sidebar, text="▶ BẮT ĐẦU KIỂM TRA", command=self.inspect, bg=BLUE, fg="white", font=("Arial",10,"bold"), height=2).pack(fill="x", padx=14, pady=4)
        tk.Button(self.sidebar, text="💾 Lưu kết quả", command=self.save_current, height=2).pack(fill="x", padx=14, pady=4)
        tk.Button(self.sidebar, text="🗂 Lịch sử kiểm tra", command=self.open_history, height=2).pack(fill="x", padx=14, pady=4)
        tk.Button(self.sidebar, text="↺ Xóa ảnh kiểm tra", command=self.clear_test, height=2).pack(fill="x", padx=14, pady=4)
        tk.Button(self.sidebar, text="Thoát", command=self.destroy, height=2).pack(fill="x", padx=14, pady=(4,12))
        tk.Label(self.sidebar, text="Mẹo camera: dùng nền đơn giản, đủ sáng, giữ góc chụp gần giống ảnh chuẩn.", bg="white", fg="#777", wraplength=235, justify="left", font=("Arial",8)).pack(anchor="w", padx=14, pady=5)

    def _build_views(self, parent):
        box = tk.LabelFrame(parent, text="Các bước xử lý", bg=BG, font=("Arial",11,"bold"))
        box.pack(fill="both", expand=True)
        self.views = {}
        names = ["Ảnh chuẩn", "Ảnh cần kiểm tra", "Ảnh xám", "Mask sản phẩm", "Phát hiện cạnh", "Ảnh khác biệt", "Kết quả"]
        positions = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0)]
        for name,(row,col) in zip(names,positions):
            card = tk.LabelFrame(box, text=name, bg="white", font=("Arial",10,"bold"), bd=1)
            card.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
            lbl = tk.Label(card, text="Chưa có ảnh", bg="white", fg="#777")
            lbl.pack(fill="both", expand=True, padx=5, pady=5)
            self.views[name] = lbl
            if name == "Kết quả":
                self.result_status = tk.Label(card, text="CHƯA KIỂM TRA", bg="white", fg="#666", font=("Arial",14,"bold"))
                self.result_status.pack(fill="x", padx=5)
                self.result_details = tk.Label(card, text="", bg="white", justify="left", anchor="w", wraplength=900)
                self.result_details.pack(fill="x", padx=10, pady=(0,5))
        # Kết quả chiếm 3 cột hàng cuối nếu có không gian
        for c in range(3): box.grid_columnconfigure(c, weight=1)
        for r in range(3): box.grid_rowconfigure(r, weight=1)
        self.views["Kết quả"].master.grid_configure(columnspan=3)

    def _build_result_panel(self, parent):
        box = tk.LabelFrame(parent, text="Kết quả kiểm tra", bg="white", font=("Arial",11,"bold"))
        box.pack(fill="x", pady=(8,0))
        left = tk.Frame(box, bg="white")
        left.pack(side="left", fill="y", padx=18, pady=12)
        self.status = tk.Label(left, text="CHƯA KIỂM TRA", bg="white", fg="#666", font=("Arial",22,"bold"))
        self.status.pack(anchor="w")
        self.summary = tk.Label(left, text="Chọn loại sản phẩm và ảnh kiểm tra.", bg="white", fg="#333", justify="left", wraplength=500)
        self.summary.pack(anchor="w", pady=(4,0))
        metrics = tk.Frame(box, bg="#f8fafc", bd=1, relief="solid")
        metrics.pack(side="right", fill="both", expand=True, padx=15, pady=12)
        self.metrics_text = tk.Text(metrics, height=6, width=55, bg="#f8fafc", relief="flat", font=("Consolas",9))
        self.metrics_text.pack(fill="both", expand=True, padx=8, pady=6)
        self.metrics_text.insert("1.0", "Chưa có chỉ số.")
        self.metrics_text.config(state="disabled")

    def cv_photo(self, img, max_w=350, max_h=210):
        if img is None: return None
        if len(img.shape)==2: img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        h,w=img.shape[:2]
        scale=min(max_w/max(w,1),max_h/max(h,1),1.0)
        im=cv2.resize(img,(max(1,int(w*scale)),max(1,int(h*scale))),interpolation=cv2.INTER_AREA)
        rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        ok,buf=cv2.imencode('.png',rgb)
        if not ok: return None
        photo=tk.PhotoImage(data=base64.b64encode(buf).decode('ascii'))
        self._photos.append(photo)
        self._photos=self._photos[-30:]
        return photo

    def set_view(self,name,img):
        p=self.cv_photo(img)
        lbl=self.views[name]
        lbl.configure(image=p,text="")
        lbl.image=p

    def _pick_image(self,title):
        return filedialog.askopenfilename(title=title,filetypes=[("Ảnh","*.jpg *.jpeg *.png *.bmp"),("Tất cả file","*.*")])

    def choose_reference(self):
        p=self._pick_image(f"Chọn ảnh chuẩn - {self.product_name.get()}")
        if not p:return
        try:
            self.reference_path=p; self.reference=resize_for_processing(load_image(p))
            self.ref_label.config(text=f"Ảnh chuẩn: {Path(p).name}")
            self.set_view("Ảnh chuẩn",self.reference); self._reset_result()
        except Exception as e: messagebox.showerror("Lỗi",str(e))

    def choose_test(self):
        p=self._pick_image(f"Chọn ảnh kiểm tra - {self.product_name.get()}")
        if not p:return
        try:
            self.test_path=p; self.test=resize_for_processing(load_image(p))
            self.test_label.config(text=f"Ảnh kiểm tra: {Path(p).name}")
            self.set_view("Ảnh cần kiểm tra",self.test); self._reset_result()
        except Exception as e: messagebox.showerror("Lỗi",str(e))

    def capture_camera(self):
        messagebox.showinfo("Camera","Nhấn SPACE để chụp, ESC để hủy. Hãy giữ sản phẩm gần cùng góc với ảnh chuẩn.")
        img=capture_from_camera(0, f"Camera - {self.product_name.get()}")
        if img is None:
            messagebox.showwarning("Camera","Không chụp được ảnh hoặc camera không khả dụng.")
            return
        self.test=resize_for_processing(img); self.test_path="camera"
        self.test_label.config(text="Ảnh kiểm tra: camera.jpg")
        self.set_view("Ảnh cần kiểm tra",self.test); self._reset_result()

    def load_demo(self):
        try:
            ref,test=DEMO[self.product_name.get()]
            self.reference_path=str(ref); self.test_path=str(test)
            self.reference=resize_for_processing(load_image(str(ref)))
            self.test=resize_for_processing(load_image(str(test)))
            self.ref_label.config(text=f"Ảnh chuẩn: {ref.name}")
            self.test_label.config(text=f"Ảnh kiểm tra: {test.name}")
            self.set_view("Ảnh chuẩn",self.reference); self.set_view("Ảnh cần kiểm tra",self.test)
            self._reset_result("Đã nạp bộ ảnh demo. Bấm BẮT ĐẦU KIỂM TRA.")
        except Exception as e:
            self._reset_result(f"Không nạp được demo: {e}")

    def inspect(self):
        if self.reference is None or self.test is None:
            messagebox.showwarning("Thiếu ảnh","Cần có ảnh chuẩn và ảnh cần kiểm tra.")
            return
        try:
            module=PRODUCTS[self.product_name.get()]
            self.inspection=module.inspect(self.reference,self.test)
            self.set_view("Ảnh xám",make_visual_gray(to_gray(self.test)))
            self.set_view("Mask sản phẩm",self.inspection.mask_image)
            self.set_view("Phát hiện cạnh",self.inspection.edge_image)
            self.set_view("Ảnh khác biệt",self.inspection.diff_image)
            self.set_view("Kết quả",self.inspection.result_image)
            self._show_result()
        except Exception as e:
            messagebox.showerror("Không thể kiểm tra",str(e))

    def _show_result(self):
        r=self.inspection
        passed = r.status == "ĐẠT"
        self.result_status.config(text="✓ SẢN PHẨM ĐẠT" if passed else "✕ SẢN PHẨM KHÔNG ĐẠT", fg=GREEN if passed else RED)
        self.result_details.config(text="Không phát hiện lỗi vượt ngưỡng." if passed else "Lỗi phát hiện:\n" + "\n".join("• " + defect for defect in r.defects))
        if r.status=="ĐẠT":
            self.status.config(text="✓ SẢN PHẨM ĐẠT",fg=GREEN)
            self.summary.config(text=f"{self.product_name.get()}: không phát hiện lỗi vượt ngưỡng demo.")
        else:
            self.status.config(text="✕ SẢN PHẨM KHÔNG ĐẠT",fg=RED)
            self.summary.config(text=f"{self.product_name.get()} - Phát hiện: " + ", ".join(r.defects))
        lines=[f"{k}: {v}" for k,v in r.metrics.items()]
        lines.append("Loại lỗi: " + ("; ".join(r.defects) if r.defects else "Không có"))
        self.metrics_text.config(state="normal"); self.metrics_text.delete("1.0","end"); self.metrics_text.insert("1.0","\n".join(lines)); self.metrics_text.config(state="disabled")

    def save_current(self):
        if self.inspection is None:
            messagebox.showwarning("Chưa có kết quả","Hãy kiểm tra trước khi lưu.")
            return
        try:
            p=save_result(self.product_name.get(),self.reference_path,self.test_path,self.inspection)
            messagebox.showinfo("Đã lưu",f"Đã lưu ảnh kết quả và lịch sử SQLite.\n\n{p}")
        except Exception as e: messagebox.showerror("Lỗi lưu",str(e))

    def clear_test(self):
        self.test=None; self.test_path=None; self.inspection=None
        self.test_label.config(text="Ảnh kiểm tra: chưa chọn")
        for name in ["Ảnh cần kiểm tra","Ảnh xám","Mask sản phẩm","Phát hiện cạnh","Ảnh khác biệt","Kết quả"]:
            self.views[name].configure(image="",text="Chưa có ảnh"); self.views[name].image=None
        self._reset_result()

    def _reset_result(self,text="Nhấn BẮT ĐẦU KIỂM TRA khi đã chọn đủ ảnh."):
        self.result_status.config(text="CHƯA KIỂM TRA", fg="#666")
        self.result_details.config(text="")
        self.views["Kết quả"].configure(image="", text="Chưa có ảnh")
        self.views["Kết quả"].image = None
        self.inspection=None; self.status.config(text="CHƯA KIỂM TRA",fg="#666"); self.summary.config(text=text)
        self.metrics_text.config(state="normal"); self.metrics_text.delete("1.0","end"); self.metrics_text.insert("1.0","Chưa có chỉ số."); self.metrics_text.config(state="disabled")

    def open_history(self):
        win=tk.Toplevel(self); win.title("Lịch sử kiểm tra 3 sản phẩm"); win.geometry("1250x550")
        cols=("id","product","file","result","defects","metrics","time")
        tree=ttk.Treeview(win,columns=cols,show="headings")
        heads={"id":"ID","product":"Sản phẩm","file":"Tên ảnh","result":"Kết quả","defects":"Loại lỗi","metrics":"Chỉ số","time":"Thời gian"}
        widths={"id":45,"product":105,"file":145,"result":95,"defects":280,"metrics":330,"time":150}
        for c in cols: tree.heading(c,text=heads[c]); tree.column(c,width=widths[c],anchor="w")
        tree.pack(fill="both",expand=True,padx=10,pady=10)

        def refresh():
            tree.delete(*tree.get_children())
            for row in get_history():
                try:
                    metrics=" | ".join(f"{k}: {v}" for k,v in json.loads(row["metrics_json"] or "{}").items())
                except Exception: metrics=row["metrics_json"] or ""
                tree.insert("","end",values=(row["id"],row["product_name"],row["image_name"],row["status"],row["defects"],metrics,row["checked_at"]))
        def selected_row():
            sel=tree.selection()
            if not sel: messagebox.showwarning("Lịch sử","Hãy chọn một dòng.",parent=win); return None
            rid=int(tree.item(sel[0],"values")[0])
            return next((r for r in get_history() if r["id"]==rid),None)
        def delete_selected():
            r=selected_row()
            if r and messagebox.askyesno("Xác nhận","Xóa bản ghi đã chọn?",parent=win): delete_inspection(r["id"]); refresh()
        def open_result():
            r=selected_row()
            if not r:return
            p=r["result_path"]
            if not p or not Path(p).exists(): messagebox.showwarning("Ảnh kết quả","Không tìm thấy file.",parent=win); return
            try:
                if platform.system()=="Windows": os.startfile(p)
                elif platform.system()=="Darwin": subprocess.Popen(["open",p])
                else: subprocess.Popen(["xdg-open",p])
            except Exception as e: messagebox.showerror("Lỗi",str(e),parent=win)
        btn=tk.Frame(win); btn.pack(fill="x",padx=10,pady=(0,10))
        def delete_all():
            if not messagebox.askyesno("Xác nhận", "Bạn có chắc muốn xóa TOÀN BỘ lịch sử kiểm tra không?\nHành động này không thể hoàn tác.", parent=win):
                return
            try:
                delete_all_inspections()
                refresh()
                messagebox.showinfo("Lịch sử", "Đã xóa toàn bộ lịch sử kiểm tra.", parent=win)
            except Exception as e:
                messagebox.showerror("Lỗi xóa lịch sử", str(e), parent=win)
        tk.Button(btn,text="Mở ảnh kết quả",command=open_result).pack(side="left",padx=4)
        tk.Button(btn,text="Xóa dòng",command=delete_selected).pack(side="left",padx=4)
        tk.Button(btn,text="Làm mới",command=refresh).pack(side="left",padx=4)
        tk.Button(btn,text="XÓA TẤT CẢ",command=delete_all).pack(side="left",padx=4)
        tk.Button(btn,text="Đóng",command=win.destroy).pack(side="right",padx=4)
        refresh()


if __name__=="__main__":
    App().mainloop()
