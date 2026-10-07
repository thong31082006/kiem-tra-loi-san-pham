# PHÂN CÔNG NHÓM 8 - BẢN 3 SẢN PHẨM

## Nguyên tắc
- Không giao ERD cho thành viên nào vì ERD được thiết kế riêng cho toàn hệ thống.
- Mỗi thành viên có một đầu việc chính rõ ràng và có phần kiểm thử riêng.
- Tất cả module phải thống nhất đầu vào là ảnh BGR của OpenCV và đầu ra là `InspectionResult`.

## 1. Nguyễn Anh Đức - Giao diện + tích hợp hệ thống
- Phụ trách `main.py`.
- Xây dựng giao diện Tkinter.
- Dropdown chọn 3 sản phẩm: Cuốc sắt, Chai nhựa, Gạch men.
- Nút chọn ảnh chuẩn, chọn ảnh kiểm tra, camera, kiểm tra, lưu kết quả, lịch sử.
- Tích hợp 3 detector và kiểm tra toàn bộ luồng chương trình.
- Quản lý nhánh main/release và ghép code cuối.

**Sản phẩm bàn giao:** `main.py`, bản chạy tích hợp, checklist kiểm thử toàn hệ thống.

## 2. Võ Văn Thông - Thu nhận ảnh + tiền xử lý + SQLite/history
- Phụ trách `image_input.py`, `image_processing.py`, `database.py`.
- Đọc ảnh Unicode trên Windows.
- Chụp ảnh từ camera.
- Resize, grayscale, Gaussian Blur, Canny, tách vật thể khỏi nền.
- Tạo và quản lý SQLite.
- Hiển thị/lấy lịch sử kiểm tra, xóa bản ghi.

**Sản phẩm bàn giao:** module đầu vào/tiền xử lý, database, test camera và test lưu lịch sử.

## 3. Nguyễn Ngọc Đương - Module kiểm tra Cuốc sắt
- Phụ trách `products/hoe_detector.py`.
- Kiểm tra mẻ/biến dạng lưỡi cuốc bằng so sánh silhouette.
- Kiểm tra gỉ sét bằng vùng màu HSV.
- Kiểm tra nứt/trầy bằng Black-hat/biên.
- Đánh dấu vùng nghi ngờ trên ảnh kết quả.
- Chuẩn bị ảnh test đạt/lỗi cho cuốc sắt.

**Sản phẩm bàn giao:** detector cuốc sắt + bộ ảnh test + kết quả test.

## 4. Giang Gia Mẫn - Module kiểm tra Chai nhựa
- Phụ trách `products/bottle_detector.py`.
- Kiểm tra thiếu/sai nắp.
- Kiểm tra méo thân/sai hình dạng bằng silhouette.
- Kiểm tra bất thường bề mặt bằng sai khác ảnh chuẩn hóa.
- Chuẩn bị ảnh test đạt/lỗi cho chai nhựa.

**Sản phẩm bàn giao:** detector chai nhựa + bộ ảnh test + kết quả test.

## 5. Nguyễn Thị Hồng Minh - Module kiểm tra Gạch men + kết quả xuất file
- Phụ trách `products/tile_detector.py` và phối hợp `result.py`.
- Kiểm tra nứt bề mặt.
- Kiểm tra mẻ cạnh/sai hình dạng.
- Kiểm tra vết bẩn/bất thường màu.
- Lưu ảnh kết quả và xác nhận dữ liệu được ghi vào SQLite.
- Chuẩn bị ảnh test đạt/lỗi cho gạch men.

**Sản phẩm bàn giao:** detector gạch men + test + kiểm tra lưu kết quả.

## Cách ghép code
1. Mỗi thành viên làm file/module của mình.
2. Test riêng trước khi merge.
3. Đức tích hợp vào `main.py`.
4. Chạy đủ 6 ca demo: 3 ảnh đạt + 3 ảnh lỗi.
5. Kiểm tra camera và SQLite/history.
