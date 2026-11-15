# Bộ AI nhận diện loại lỗi

HOG + SVM · phân tích 6 nhóm

Nhận diện loại lỗi chính trên mẫu chai Ron 88. Nhãn hỏng, lệch và thiếu được nguồn gộp chung. Chưa kiểm chứng nhiều lỗi đồng thời, hãng chai khác hoặc lỗi bị khuất. Chưa hỗ trợ móp chai và nước đục; ảnh không xác nhận được độ kín thực tế của nắp.

Nhãn loại lỗi chỉ dùng làm mục tiêu huấn luyện/đánh giá, không đưa vào dự đoán. Chọn cấu hình và ngưỡng bằng Validation trước khi xem Test; Test là đánh giá phát triển. Mỗi ảnh nguồn có một loại lỗi chính; chưa đánh giá nhiều lỗi đồng thời.

| Loại | Train | Validation | Test | Precision Test | Recall Test | F1 Test | Bật nhận diện |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Không thấy lỗi trong các nhóm đã học | 374 | 67 | 68 | 83.75% | 98.53% | 90.54% | Có |
| Thiếu mức nước | 142 | 34 | 34 | 100.00% | 100.00% | 100.00% | Có |
| Thiếu nắp | 60 | 34 | 34 | 100.00% | 100.00% | 100.00% | Có |
| Nắp lỏng / hỏng | 68 | 27 | 34 | 100.00% | 97.06% | 98.51% | Có |
| Dị vật nhìn thấy | 68 | 34 | 27 | 100.00% | 100.00% | 100.00% | Có |
| Nhãn hỏng / lệch / thiếu | 17 | 34 | 34 | 0.00% | 0.00% | 0.00% | Chưa đạt điều kiện Validation |

Nhãn hỏng/lệch/thiếu là một nhóm chung theo annotation nguồn. Chưa hỗ trợ móp chai và nước đục. Không suy ra lực siết hoặc độ kín của nắp chỉ từ ảnh.

Run: `20261115_130628_717141`; manifest SHA256: `825c4e5158fcffcb58cb417a0cb5aee4172bbcc48a288685cc491ae7d409c63a`. Quyết định trước Test lưu tại `selection_frozen.json`.