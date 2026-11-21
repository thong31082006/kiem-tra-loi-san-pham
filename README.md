# Kiểm tra lỗi chai nhựa — Nhóm 8

Ứng dụng web chạy trên máy tính, dùng **HOG + SVM** để kiểm tra ảnh chai nhựa. Bộ định vị SSDLite tìm vùng chai trước khi trích xuất HOG. Repository có sẵn mô hình đã huấn luyện và trọng số định vị để kiểm tra ảnh sau khi cài thư viện.

## Chức năng

- Kiểm tra một ảnh hoặc một lô ảnh; trả về đạt, không đạt hoặc chưa kết luận.
- Phân loại các nhóm lỗi đã học: thiếu nước, thiếu nắp, nắp lỏng/hỏng, dị vật và nhãn hỏng/lệch/thiếu.
- Lưu lịch sử, xem lại kết quả và ghi nhận đánh giá của người dùng.
- Xem thư viện dữ liệu, huấn luyện HOG + SVM và tải báo cáo khi đã chuẩn bị bộ dữ liệu.

Mô hình thử nghiệm dành cho dữ liệu chai Ron 88. Nên chụp riêng một chai đứng, thấy rõ nắp và đáy. Chưa kiểm chứng cho mọi loại chai, góc nhìn mới hoặc lỗi bị che khuất; chưa hỗ trợ kết luận móp chai và nước đục. Các ngưỡng chưa đạt độ tin cậy sẽ cho kết quả chưa xác định.

## Cài và chạy trên Windows

Cài **Python 3.13, 64-bit** rồi tải hoặc clone repository:

```powershell
git clone https://github.com/thong31082006/kiem-tra-loi-san-pham.git
cd kiem-tra-loi-san-pham
```

Chạy `setup_windows.bat` để tạo môi trường `.venv` và cài thư viện, sau đó chạy `run_app.bat`. Trang web mặc định mở tại <http://127.0.0.1:8080>.

Hoặc cài và chạy bằng PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Ứng dụng có thể chạy bằng CPU; không bắt buộc có GPU NVIDIA. Lần cài thư viện cần kết nối mạng. Để chạy máy chủ mà không tự mở trình duyệt:

```powershell
.\.venv\Scripts\python.exe main.py --no-browser --port 8080
```

## Bộ dữ liệu và huấn luyện

Ảnh, nhãn gốc, ảnh đã cắt và dữ liệu lịch sử được loại khỏi Git bằng `.gitignore`. Các mô hình cần để suy luận, ảnh demo và thông tin nguồn dữ liệu được giữ trong repository. Kiểm tra ảnh tải lên không yêu cầu tải bộ dữ liệu huấn luyện.

Nguồn dữ liệu được ghi trong `data/datasets/bottle_defects/source.json`. Để tải dữ liệu công khai và chuẩn bị thư viện/huấn luyện, chạy lần lượt:

```powershell
.\.venv\Scripts\python.exe scripts/download_defect_dataset.py
.\.venv\Scripts\python.exe scripts/prepare_training_data.py
.\.venv\Scripts\python.exe scripts/prepare_bottle_crops.py
```

Quá trình này tạo `data/training/manifest.json` và ảnh vùng chai. Sau đó có thể huấn luyện trên giao diện web hoặc chạy:

```powershell
.\.venv\Scripts\python.exe scripts/train_hog_svm.py
```

Huấn luyện chọn tham số bằng tập Train/Validation và dùng Test để đánh giá. Mô hình hiện hành được chỉ định bởi `models/hog_svm/latest.json`; báo cáo nằm cùng thư mục của lần huấn luyện tương ứng.

## Cấu trúc chính

| Đường dẫn | Nội dung |
| --- | --- |
| `main.py`, `web_api.py` | Khởi động web và API FastAPI |
| `web/svm.*` | Giao diện web hiện tại |
| `ml/` | Định vị chai, đặc trưng HOG, suy luận SVM và đánh giá |
| `web_dataset.py`, `web_training.py` | Thư viện dữ liệu và huấn luyện |
| `web_history.py` | Lịch sử kiểm tra và đánh giá |
| `models/hog_svm/` | Mô hình SVM và báo cáo |
| `models/pretrained/checkpoints/` | Trọng số bộ định vị chai |
| `scripts/` | Tải, chuẩn bị dữ liệu, huấn luyện và xuất báo cáo |

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py" -v
```

Một số kiểm tra suy luận thực cần bộ dữ liệu đã chuẩn bị. Kiểm tra giao diện JavaScript có thể chạy với Node.js:

```powershell
node --test test_svm_ui.cjs
```
