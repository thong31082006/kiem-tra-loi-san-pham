# Kiểm tra chai nhựa — Nhóm 8

Ứng dụng web phân loại một ảnh chai nhựa bằng mô hình đã huấn luyện. Người dùng chọn ảnh từ máy, chụp camera hoặc chọn ảnh trong thư viện, sau đó xem kết quả và lịch sử.

## Chạy bằng F5

1. Mở thư mục ngoài cùng `KiemTraLoiSanPham_3SanPham_Nhom8` bằng VS Code.
2. Cài extension **Python** và **Python Debugger** của Microsoft.
3. Chạy `setup_windows.bat` trong thư mục mã nguồn nếu chưa có môi trường. Script dùng Python 3.13, tạo `.venv-ml` ở thư mục ngoài cùng và cài các thư viện cần thiết.
4. Trong **Run and Debug**, chọn **Chay web kiem tra chai nhua (F5)** rồi nhấn **F5**.
5. Chương trình mở Chrome tại `http://127.0.0.1:8080`. Giữ terminal chạy trong lúc sử dụng; nhấn **Ctrl+C** để dừng.

Có thể mở `run_app.bat`, hoặc từ thư mục mã nguồn chạy:

```powershell
..\.venv-ml\Scripts\python.exe main.py
```

Nếu cổng 8080 đang được dùng, thêm `--port 8081`. Thêm `--no-browser` để chỉ mở máy chủ và tự truy cập địa chỉ trong trình duyệt.

## Sử dụng

1. Trong tab **Kiểm tra**, chọn một ảnh hoặc bấm **Chụp từ camera**. Có thể kéo ảnh vào vùng xem trước.
2. Chọn mô hình và bấm **Kiểm tra ảnh**.
3. Xem nhãn **Bình thường**, **Có dấu hiệu lỗi** hoặc **Chưa đủ tin cậy**, cùng điểm mô hình và thời gian xử lý.
4. Mở **Lịch sử** để xem lại. Mỗi lần kiểm tra được lưu tự động; có thể xóa từng bản ghi.

Tab **Thư viện ảnh** cho phép lọc Train/Validation/Test và nhãn nguồn. Nhãn dưới ảnh là nhãn dữ liệu; mô hình chỉ nhận nội dung ảnh khi dự đoán.

Camera yêu cầu quyền truy cập trong trình duyệt. Chụp rõ toàn bộ chai, gồm nắp và đáy. Nếu vùng cần kiểm tra bị che, hãy đổi góc và chụp lại.

## Ba mô hình

| Mô hình | Đầu vào xử lý | Cách huấn luyện |
| --- | --- | --- |
| HOG + SVM | HOG và histogram HSV của vùng chai | Chuẩn hóa đặc trưng, SVM RBF có trọng số cân bằng lớp |
| HOG + Random Forest | Cùng bộ đặc trưng HOG và HSV của vùng chai | 160 cây, độ sâu tối đa 12, trọng số cân bằng lớp |
| MobileNetV3 Small | Vùng chai RGB 224 × 224 | Chuyển giao từ ImageNet, huấn luyện phần phân loại rồi điều chỉnh hai khối đặc trưng cuối |

Chương trình tự tìm chai trong ảnh bằng SSDLite đã huấn luyện sẵn trên COCO rồi cắt vùng chai cho mô hình phân loại. Dữ liệu huấn luyện cũng đi qua đúng bộ định vị/cắt ảnh này, được lưu PNG không mất dữ liệu. Annotation nguồn dùng xác định nhãn lỗi, không dùng làm vị trí cắt trong phiên bản cuối. Ảnh cần có một chai nhìn thấy rõ; nếu không tìm thấy chai hoặc có nhiều chai, chương trình yêu cầu chọn/chụp lại.

Tab **Mô hình** hiển thị Accuracy, Precision, Recall và F1 lấy từ báo cáo huấn luyện đã lưu. Precision, Recall và F1 dùng `defect` làm lớp dương. Các chỉ số được tính cho dự đoán hai lớp trước bước từ chối do thiếu tin cậy. Điểm của từng ảnh chưa được hiệu chuẩn thành xác suất đúng thực tế; kết quả dưới ngưỡng tin cậy 0,65 được hiển thị là **Chưa đủ tin cậy**.

Số liệu và ma trận nhầm lẫn nằm trong [BAO_CAO_MO_HINH.md](BAO_CAO_MO_HINH.md). Báo cáo tách kết quả phân loại hai lớp và kết quả toàn bộ luồng web, gồm cả trường hợp chưa đủ tin cậy. HOG + SVM là lựa chọn mặc định cho bộ dữ liệu thử nghiệm này; có thể đổi sang hai mô hình còn lại trên web. Đây là số liệu phát triển trong phạm vi bộ nguồn.

Các mô hình được đánh giá trong phạm vi mẫu chai Ron 88 và điều kiện chụp của bộ dữ liệu. Tập Test hiện tại đã được tham khảo trong quá trình phát triển, nên số liệu là đánh giá phát triển, chưa phải kiểm tra mù độc lập. Cần thu thập chai và phiên chụp mới để đánh giá trước khi sử dụng trong thực tế.

Kế hoạch bổ sung ảnh thật từ nhiều góc, cách gán nhãn lỗi bị khuất và chia theo ID chai vật lý nằm trong [THU_THAP_DU_LIEU.md](THU_THAP_DU_LIEU.md).

## Dữ liệu và huấn luyện lại

Bộ dữ liệu chính là [Water Bottle Defect Level Detection Dataset trên Kaggle](https://www.kaggle.com/datasets/jihadakbr/water-bottle-defect-level-detection-dataset), gồm ảnh chai Ron 88 và annotation lỗi. Nguồn, phiên bản và SHA256 được lưu tại `data/datasets/bottle_defects/source.json`.

Manifest cuối tại `data/training/manifest.json` có 1.190 ảnh: 729 Train, 230 Validation và 231 Test. Sau khi chia khối có 1.193 ảnh; bộ định vị từ chối thêm ba ảnh Train/Validation và ghi lý do trong manifest. Các bản ghi giữ đường dẫn ảnh gốc, vùng chai tự cắt, nhãn, nhóm và split. Nguồn chưa có ID chai vật lý nên vẫn có thể còn ảnh gần giống giữa các nhóm. Việc phân loại dùng pixel ảnh, không dùng tên tệp hoặc nhãn nguồn làm đặc trưng. Chi tiết trong [DATASET.md](DATASET.md).

Từ thư mục mã nguồn:

```powershell
..\.venv-ml\Scripts\python.exe scripts\download_defect_dataset.py
..\.venv-ml\Scripts\python.exe scripts\prepare_training_data.py
..\.venv-ml\Scripts\python.exe scripts\prepare_bottle_crops.py
..\.venv-ml\Scripts\python.exe scripts\train_models.py --epochs 8 --batch-size 8 --device auto
..\.venv-ml\Scripts\python.exe scripts\evaluate_web_pipeline.py
..\.venv-ml\Scripts\python.exe scripts\export_report.py
```

Hoặc bấm **Huấn luyện lại** trong tab **Mô hình** để huấn luyện và đánh giá cả ba mô hình trên dữ liệu đã chuẩn bị. Trạng thái được cập nhật trong khi chạy. `--device auto` dùng CUDA nếu có, hoặc CPU. Epoch tốt nhất và ngưỡng phân loại chọn bằng Validation; Test là bộ đánh giá phát triển, không dùng chọn ngưỡng.

`evaluate_web_pipeline.py` chạy toàn bộ ảnh Test bằng `ModelService` thực tế và ghi `end_to_end.json`, không tạo lịch sử kiểm tra. Báo cáo tính ảnh chưa đủ tin cậy hoặc không định vị được chai là chưa có kết luận đúng, và ghi tỷ lệ ảnh được kết luận (Coverage).

Mỗi lần huấn luyện ghi vào `models/trained/<run_id>/`. Chỉ khi cả ba mô hình hoàn tất, `latest.json` mới được chuyển sang lần chạy mới. Nếu huấn luyện lỗi, bộ mô hình trước đó được giữ để tiếp tục sử dụng. Kết quả thực tế và phạm vi kiểm tra được ghi trong [TEST_LOG.txt](TEST_LOG.txt).

## Kiểm tra chương trình

```powershell
..\.venv-ml\Scripts\python.exe test_web_app.py
..\.venv-ml\Scripts\python.exe test_ml_contract.py
```

Bài kiểm tra API dùng lịch sử và ảnh trong thư mục tạm, cùng bộ mô phỏng dịch vụ dự đoán để kiểm tra luồng HTTP. Việc kiểm tra API không thay thế đánh giá mô hình trên dữ liệu Test.

## Cấu trúc chính

- `main.py`: mở máy chủ cục bộ và trình duyệt.
- `web/`: HTML, CSS và JavaScript giao diện.
- `web_api.py`: API ảnh, dự đoán, mô hình, thư viện và huấn luyện.
- `web_history.py`: lịch sử dự đoán tại `data/web/history.db`.
- `ml/features.py`: đọc ảnh, HOG/HSV và biến đổi ảnh cho CNN.
- `ml/train.py`: huấn luyện, chọn ngưỡng và đánh giá.
- `ml/inference.py`: nạp mô hình và dự đoán từ một ảnh.
- `scripts/`: tải dữ liệu, chuẩn bị manifest và chạy huấn luyện.
- `models/trained/`: mô hình và báo cáo các lần chạy.
- `ERD.md`: cấu trúc lưu trữ.
- `PHAN_CONG_NHOM_MOI.md`: phân công nhóm.

Cơ sở dữ liệu cũ `data/inspection.db` được giữ nguyên. Giao diện Tkinter trước đây nằm trong `desktop_legacy.py`; ứng dụng chính hiện tại là web.
