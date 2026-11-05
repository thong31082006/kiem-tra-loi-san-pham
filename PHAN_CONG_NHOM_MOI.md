# Phân công Nhóm 8 — Web kiểm tra chai nhựa

Cả nhóm thực hiện phân loại một ảnh chai nhựa bằng ba mô hình: HOG + SVM, HOG + Random Forest và MobileNetV3 Small. Dùng chung manifest và cách tách Train/Validation/Test khi so sánh kết quả.

## 1. Nguyễn Anh Đức — Giao diện và tích hợp

- Phụ trách `web/`, `main.py` và cấu hình chạy F5.
- Xây dựng chọn ảnh, kéo thả, camera và hiển thị kết quả.
- Tích hợp API thư viện, mô hình, huấn luyện và lịch sử.
- Kiểm tra trạng thái chưa huấn luyện, đang xử lý và lỗi kết nối.

**Bàn giao:** giao diện web chạy trên Chrome và hướng dẫn sử dụng.

## 2. Võ Văn Thông — Dữ liệu và tiền xử lý

- Phụ trách script tải dữ liệu, chuẩn bị manifest và `ml/features.py`.
- Ghi nguồn, phiên bản, giấy phép và SHA256 bộ Kaggle.
- Đọc annotation, ánh xạ nhãn bình thường/có lỗi và phân nhóm ảnh.
- Tách nhóm giữa Train/Validation/Test, kiểm tra ảnh lỗi hoặc trùng.
- Dùng cùng cách đọc ảnh và biến đổi đặc trưng cho huấn luyện/dự đoán.

**Bàn giao:** dữ liệu đã kiểm tra, manifest và module đặc trưng.

## 3. Nguyễn Ngọc Đương — SVM và Random Forest

- Phụ trách hai mô hình cổ điển trong `ml/train.py`.
- Chuẩn hóa HOG/HSV cho SVM và cấu hình Random Forest.
- Cân bằng lớp, chọn ngưỡng bằng Validation.
- Lưu artifact, báo cáo Precision/Recall/F1 và ma trận nhầm lẫn trên Test.

**Bàn giao:** hai mô hình đã huấn luyện và cấu hình chạy lại.

## 4. Giang Gia Mẫn — MobileNetV3

- Phụ trách nhánh CNN trong `ml/train.py` và nạp mô hình trong `ml/inference.py`.
- Chuyển giao từ ImageNet, tăng cường dữ liệu chỉ cho Train.
- Theo dõi loss và F1 Validation, chọn epoch tốt nhất.
- Kiểm tra suy luận từ pixel ảnh và cách xử lý kết quả chưa đủ tin cậy.

**Bàn giao:** checkpoint CNN, đường học và kết quả đánh giá.

## 5. Nguyễn Thị Hồng Minh — API, lịch sử và kiểm thử

- Phụ trách `web_api.py`, `web_history.py` và `test_web_app.py`.
- Kiểm tra ảnh đầu vào, gọi mô hình và lưu lịch sử dự đoán.
- Lưu ảnh an toàn theo UUID, xem và xóa từng bản ghi.
- Kiểm tra API bằng dữ liệu tạm; tổng hợp bảng so sánh mô hình thực tế.
- Cập nhật cấu trúc lưu trữ và nhật ký kiểm tra.

**Bàn giao:** API, SQLite lịch sử và kết quả kiểm thử.

## Ghép mã và đánh giá

1. Chốt manifest và giữ nguyên Test khi lựa chọn mô hình/ngưỡng.
2. Huấn luyện cả ba mô hình với cùng split; ghi cấu hình và seed.
3. Đánh giá trên Test sau khi chọn xong bằng Validation.
4. Kiểm tra luồng chọn một ảnh, camera, dự đoán và lịch sử trên web.
5. Chạy kiểm tra API tách biệt lịch sử thật; ghi kết quả vào `TEST_LOG.txt`.
6. Báo cáo đúng phạm vi mẫu chai và góc chụp đã có trong dữ liệu.
