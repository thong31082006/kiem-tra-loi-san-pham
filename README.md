# CHƯƠNG TRÌNH KIỂM TRA LỖI SẢN PHẨM - NHÓM 8

## Phạm vi chính thức
Ứng dụng kiểm tra 3 loại sản phẩm:
1. **Cuốc sắt**: gỉ sét, mẻ/biến dạng, nứt/trầy.
2. **Chai nhựa**: thiếu/sai nắp, méo thân, bất thường bề mặt.
3. **Gạch men**: nứt, mẻ cạnh/sai hình dạng, vết bẩn/bất thường màu.

## Công nghệ
- Giao diện: Tkinter
- Ngôn ngữ: Python
- Xử lý ảnh: OpenCV + NumPy
- Database: SQLite
- IDE: PyCharm
- Quản lý mã nguồn: Git + GitHub

## Luồng chương trình
Chọn loại sản phẩm -> nạp/chọn ảnh chuẩn -> chọn ảnh hoặc chụp camera -> tiền xử lý -> chạy detector tương ứng -> ĐẠT/KHÔNG ĐẠT -> lưu ảnh kết quả -> lưu SQLite -> xem lịch sử.

## Cách chạy
1. Mở thư mục project bằng PyCharm.
2. Mở Terminal và chạy: `pip install -r requirements.txt`
3. Chạy file `main.py`.
4. Chọn loại sản phẩm ở combobox.
5. App tự nạp bộ demo tương ứng. Bấm **BẮT ĐẦU KIỂM TRA**.
6. Muốn dùng ảnh thật: bấm **Chọn ảnh chuẩn** và **Chọn ảnh cần kiểm tra**, hoặc dùng **Chụp từ CAMERA** cho ảnh cần kiểm tra.

## Camera
Có thể chụp trực tiếp một ảnh cần kiểm tra. Ứng dụng vẫn cần ảnh chuẩn của đúng loại sản phẩm để so sánh. Khi cửa sổ camera mở:
- SPACE: chụp
- ESC: hủy

## Điều kiện ảnh thật nên giữ ổn định
- Cùng loại sản phẩm với ảnh chuẩn.
- Góc chụp gần giống nhau.
- Nền sáng/đơn giản, tương phản với vật thể.
- Khoảng cách camera gần tương đương.
- Đủ sáng, hạn chế bóng mạnh.

## Ảnh demo
Ảnh demo trong project được tạo để kiểm thử thuật toán, không phải dataset công nghiệp. Mỗi loại có ảnh chuẩn, ảnh đạt và ảnh lỗi.

## File quan trọng
- `main.py`: giao diện và điều phối.
- `products/hoe_detector.py`: cuốc sắt.
- `products/bottle_detector.py`: chai nhựa.
- `products/tile_detector.py`: gạch men.
- `image_input.py`: đọc ảnh/camera.
- `image_processing.py`: xử lý ảnh dùng chung.
- `database.py`: SQLite.
- `result.py`: lưu kết quả.
- `ERD.md`: ERD chi tiết.
- `PHAN_CONG_NHOM_MOI.md`: phân công 5 thành viên.

## Lưu ý học thuật
Đây là prototype phục vụ môn Xử lý ảnh & Thị giác máy tính. Các ngưỡng được thiết kế để minh họa bằng bộ ảnh demo. Muốn đánh giá chính xác trên ảnh thực tế cần thu thập ảnh cùng điều kiện và hiệu chỉnh ngưỡng cho từng sản phẩm.
