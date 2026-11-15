# Thu thập ảnh chai nhựa từ nhiều góc

Bộ Kaggle hiện tại chỉ hỗ trợ thử nghiệm trong một điều kiện chụp. Để đánh giá bằng camera và nhiều góc nhìn, cần bổ sung ảnh thật có ID chai vật lý và các lần chụp độc lập.

## Bắt đầu với một mẫu chai

Chọn một mẫu chai/SKU, thống nhất loại lỗi cần phát hiện và điều kiện ảnh được chấp nhận. Lấy nhiều chai vật lý khác nhau, gồm chai bình thường và các trường hợp lỗi. Giữ bản ghi chai trước và sau khi tạo lỗi dưới cùng một `bottle_id`.

Với mỗi chai, chụp các mặt quanh thân, góc nghiêng, phần nắp và phần đáy khi liên quan. Ghi góc nhìn, camera, nền, ánh sáng và phiên chụp. Ảnh bình thường và ảnh lỗi cần có các điều kiện tương ứng, tránh để mô hình học riêng nền hoặc góc chụp của một lớp.

## Nhãn theo phần nhìn thấy trong ảnh

Ghi thông tin tối thiểu:

| Trường | Ví dụ/ý nghĩa |
| --- | --- |
| `bottle_id` | ID cố định cho một chai vật lý |
| `session_id` | Phiên chụp, camera và điều kiện sáng |
| `view` | Mặt trước, bên, sau, góc nghiêng, nắp, đáy |
| `defect_type` | Thiếu nắp, nắp lỏng, dị vật, thiếu nước, nhãn hỏng |
| `defect_visible` | Lỗi có nhìn thấy trong khung hình hay không |
| `label` | Bình thường, lỗi nhìn thấy, hoặc chưa đủ thông tin |

Một chai có lỗi nhưng lỗi nằm ở mặt khuất không đồng nghĩa ảnh đang nhìn thấy có bằng chứng lỗi. Đánh dấu ảnh đó là chưa đủ thông tin hoặc yêu cầu ảnh từ góc khác. Khi sử dụng, hướng dẫn người dùng chụp lại hoặc chụp thêm mặt liên quan; chưa kết luận chỉ từ một ảnh bị che.

## Chia tập trước khi tăng cường ảnh

- Mọi góc chụp và mọi ảnh của cùng `bottle_id` thuộc một tập duy nhất: Train, Validation hoặc Test.
- Ảnh trước/sau khi tạo lỗi, ảnh liên tiếp và bản chỉnh sửa của cùng chai cũng phải theo cùng nhóm.
- Giữ một phần chai và phiên chụp mới để kiểm tra cuối; không dùng chúng để chọn epoch, ngưỡng hoặc chỉnh giao diện kết luận.
- Kiểm tra ảnh trùng và gần trùng trước khi chia. Chỉ tăng cường dữ liệu trong Train.
- Giữ đủ trường hợp bình thường và từng loại lỗi/góc nhìn trong các tập; ghi số chai vật lý riêng với số ảnh.

## Đánh giá và đưa vào ứng dụng

Dùng cùng split để so sánh ba mô hình. Ngoài Accuracy/Precision/Recall/F1 tổng thể, xem riêng lỗi bị bỏ sót theo loại lỗi, góc chụp và điều kiện sáng. Ghi cả tỷ lệ ảnh bị từ chối do thiếu tin cậy.

Chỉ mở rộng sang mẫu chai/SKU khác sau khi có dữ liệu và đánh giá riêng cho mẫu đó. Kết quả của một mẫu chai hoặc các góc đã chụp không chứng minh khả năng kiểm tra mọi chai, mọi góc nhìn hoặc lỗi nằm ngoài khung hình.
