# Kết quả huấn luyện và kiểm tra chai nhựa

Lần chạy `20261105_071747_137048`, hoàn tất 05/11/2026 14:18 (giờ Việt Nam).

Mỗi yêu cầu dùng một ảnh. SSDLite COCO đã huấn luyện sẵn tìm chai; ba bộ phân loại chất lượng được huấn luyện bằng dữ liệu của dự án. Không dùng ảnh chuẩn hoặc so sánh với ảnh chuẩn.

## Kết quả toàn bộ luồng web

Đã chạy 231 ảnh Test × 3 mô hình = 693 dự đoán thực tế, trong 52.54 giây. Không tạo lịch sử web khi đánh giá.

| Mô hình | Đúng / tổng ảnh | Precision lỗi | Recall lỗi | F1 lỗi | Có kết luận | Chưa kết luận |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HOG + SVM | 91,77% | 98,66% | 90,18% | 94,23% | 92,64% | 17 |
| HOG + Random Forest | 77,06% | 100,00% | 71,78% | 83,57% | 77,49% | 52 |
| MobileNetV3 Small | 73,16% | 99,08% | 66,26% | 79,41% | 87,88% | 28 |

Đánh giá toàn ảnh qua định vị chai tự động và phân loại thực tế; không lưu lịch sử web. Accuracy tính đúng/tổng ảnh, uncertain và lỗi định vị không được tính đúng. Coverage là tỷ lệ nhận nhãn normal/defect. Recall lớp defect tính cả ảnh bị từ chối là bỏ sót. Test cùng nguồn đã dùng trong phát triển, chưa phải kiểm tra mù trên chai/phiên chụp độc lập.

Ma trận: hàng là nhãn thật; cột là nhãn được trả cho người dùng.

| Mô hình | Nhãn thật | Bình thường | Có lỗi | Chưa đủ tin cậy |
| --- | --- | ---: | ---: | ---: |
| HOG + SVM | Bình thường | 65 | 2 | 1 |
| HOG + SVM | Có lỗi | 0 | 147 | 16 |
| HOG + Random Forest | Bình thường | 61 | 0 | 7 |
| HOG + Random Forest | Có lỗi | 1 | 117 | 45 |
| MobileNetV3 Small | Bình thường | 61 | 1 | 6 |
| MobileNetV3 Small | Có lỗi | 33 | 108 | 22 |

## Bộ phân loại trước bước từ chối thiếu tin cậy

Đầu vào cũng là vùng chai do bộ định vị tự tìm. Bảng này chưa áp dụng việc yêu cầu chụp lại khi điểm dưới 0,65.

| Mô hình | Accuracy | Precision lỗi | Recall lỗi | F1 lỗi | Ngưỡng phân loại |
| --- | ---: | ---: | ---: | ---: | ---: |
| HOG + SVM | 98,70% | 98,19% | 100,00% | 99,09% | 0.100 |
| HOG + Random Forest | 96,54% | 95,86% | 99,39% | 97,59% | 0.300 |
| MobileNetV3 Small | 82,68% | 94,89% | 79,75% | 86,67% | 0.250 |

## Dữ liệu và cách huấn luyện

- 1190 ảnh: 729 Train, 230 Validation, 231 Test.
- Khối 40 chỉ số ảnh được giữ cùng tập; bỏ ba khung ở mỗi cạnh. Seed 42. Nhãn hiệu khác không bị coi là lỗi chất lượng.
- Tự tìm và lưu vùng chai không mất dữ liệu trước khi huấn luyện; lúc sử dụng áp dụng đúng cách xử lý này. Tên tệp, annotation và nhãn thư viện không đi vào mô hình dự đoán.
- HOG + histogram HSV; SVM RBF C=2; Random Forest 160 cây, sâu tối đa 12. Cân bằng lớp từ dữ liệu Train.
- MobileNetV3 Small học chuyển giao từ ImageNet; batch 8, 8 epoch trên cuda; chọn epoch 5 bằng Validation.
- Chuẩn bị đặc trưng, huấn luyện và đánh giá bộ phân loại trong 55.72 giây; chưa gồm chuẩn bị cache vùng chai và đánh giá toàn luồng web.

Đường học của MobileNet (F1 Validation tại ngưỡng 0,5, trước khi chọn ngưỡng cuối):

| Epoch | Train loss | Validation loss | Validation F1 |
| ---: | ---: | ---: | ---: |
| 1 | 0.5676 | 0.5424 | 74,51% |
| 2 | 0.3909 | 0.5926 | 82,83% |
| 3 | 0.4135 | 0.8104 | 72,83% |
| 4 | 0.3092 | 0.5806 | 82,17% |
| 5 | 0.2316 | 0.7563 | 85,71% |
| 6 | 0.2435 | 0.6885 | 85,43% |
| 7 | 0.2017 | 0.8483 | 85,02% |
| 8 | 0.1920 | 0.9324 | 85,21% |

## Phạm vi kết luận

Dữ liệu có một mẫu chai Ron 88 và bối cảnh chụp giới hạn. Nguồn không có ID chai vật lý; ảnh gần giống có thể còn xuất hiện ở các khối khác nhau. Tập Test đã được xem trong phát triển, bao gồm việc sửa chênh lệch cách cắt ảnh khi học và sử dụng. Đây là số liệu phát triển, chưa phải kiểm tra mù trên chai hoặc phiên chụp độc lập.

Không suy ra khả năng nhận đúng từ mọi góc chụp, ảnh ngoài nguồn hoặc lỗi bị che khuất từ các chỉ số trên. Cần bộ ảnh thật có nhiều góc và các chai/phiên chụp độc lập để đánh giá tiếp. Điểm mô hình chưa được xác nhận là xác suất đúng thực tế.

Thời gian phân loại trên vùng chai, không gồm đọc tệp hoặc bộ định vị chai. CNN đo trên GPU huấn luyện, mô hình cổ điển trên CPU; web ghi thời gian toàn bộ yêu cầu thực tế trên CPU.

Camera vật lý chưa được kiểm thử. Xem [kế hoạch thu thập dữ liệu](THU_THAP_DU_LIEU.md) và [nguồn dữ liệu](DATASET.md).

Bằng chứng gốc: [cấu hình và chỉ số](models/trained/20261105_071747_137048/report.json), [toàn bộ dự đoán web](models/trained/20261105_071747_137048/end_to_end.json). Các lần chạy cũ được giữ riêng; `latest.json` xác định mô hình đang dùng.
