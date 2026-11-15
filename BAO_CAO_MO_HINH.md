# Kết quả huấn luyện và kiểm tra chai nhựa

Lần chạy `20261115_120255_246951`, hoàn tất 15/11/2026 19:05 (giờ Việt Nam).

Mỗi yêu cầu dùng một ảnh. SSDLite COCO đã huấn luyện sẵn tìm chai; ba bộ phân loại chất lượng được huấn luyện bằng dữ liệu của dự án. Không dùng ảnh chuẩn hoặc so sánh với ảnh chuẩn.

## Kết quả toàn bộ luồng web

Đã chạy 231 ảnh Test × 3 mô hình = 693 dự đoán thực tế, trong 49.13 giây. Không tạo lịch sử web khi đánh giá.

| Mô hình | Đúng / tổng ảnh | Precision lỗi | Recall lỗi | F1 lỗi | Có kết luận | Chưa kết luận |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HOG + SVM | 94,81% | 99,35% | 93,25% | 96,20% | 98,27% | 4 |
| HOG + Random Forest | 78,79% | 100,00% | 71,78% | 83,57% | 80,09% | 46 |
| MobileNetV3 Small | 87,88% | 99,28% | 84,05% | 91,03% | 97,84% | 5 |

Đánh giá toàn ảnh qua định vị chai tự động và phân loại thực tế; không lưu lịch sử web. Accuracy tính đúng/tổng ảnh, uncertain và lỗi định vị không được tính đúng. Coverage là tỷ lệ nhận nhãn normal/defect. Recall lớp defect tính cả ảnh bị từ chối là bỏ sót. Test cùng nguồn đã dùng trong phát triển, chưa phải kiểm tra mù trên chai/phiên chụp độc lập.

Ma trận: hàng là nhãn thật; cột là nhãn được trả cho người dùng.

| Mô hình | Nhãn thật | Bình thường | Có lỗi | Chưa đủ tin cậy |
| --- | --- | ---: | ---: | ---: |
| HOG + SVM | Bình thường | 67 | 1 | 0 |
| HOG + SVM | Có lỗi | 7 | 152 | 4 |
| HOG + Random Forest | Bình thường | 65 | 0 | 3 |
| HOG + Random Forest | Có lỗi | 3 | 117 | 43 |
| MobileNetV3 Small | Bình thường | 66 | 1 | 1 |
| MobileNetV3 Small | Có lỗi | 22 | 137 | 4 |

## Bộ phân loại trước bước từ chối thiếu tin cậy

Đầu vào cũng là vùng chai do bộ định vị tự tìm. Bảng này chưa áp dụng việc yêu cầu chụp lại khi điểm dưới 0,65.

| Mô hình | Accuracy | Precision lỗi | Recall lỗi | F1 lỗi | Ngưỡng phân loại |
| --- | ---: | ---: | ---: | ---: | ---: |
| HOG + SVM | 96,54% | 99,36% | 95,71% | 97,50% | 0.500 |
| HOG + Random Forest | 92,64% | 99,32% | 90,18% | 94,53% | 0.500 |
| MobileNetV3 Small | 89,18% | 99,29% | 85,28% | 91,75% | 0.500 |

## Dữ liệu và cách huấn luyện

- 1190 ảnh: 729 Train, 230 Validation, 231 Test.
- Khối 40 chỉ số ảnh được giữ cùng tập; bỏ ba khung ở mỗi cạnh. Seed 42. Nhãn hiệu khác không bị coi là lỗi chất lượng.
- Tự tìm và lưu vùng chai không mất dữ liệu trước khi huấn luyện; lúc sử dụng áp dụng đúng cách xử lý này. Tên tệp, annotation và nhãn thư viện không đi vào mô hình dự đoán.
- HOG + histogram HSV; SVM RBF: {'svc__C': 8.0, 'svc__gamma': 5e-05}; Random Forest: {'max_depth': 12, 'min_samples_leaf': 1, 'n_estimators': 160}. Cân bằng lớp từ dữ liệu Train.
- MobileNetV3 Small học chuyển giao từ ImageNet; batch 8, 20 epoch trên cuda; chọn epoch 16 bằng Validation.
- Chuẩn bị đặc trưng, huấn luyện và đánh giá bộ phân loại trong 157.03 giây; chưa gồm chuẩn bị cache vùng chai và đánh giá toàn luồng web.

Đường học của MobileNet (F1 hai lớp trên Validation, tại ngưỡng ghi theo từng epoch):

| Epoch | Train loss | Validation loss | Validation F1 | Ngưỡng |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5676 | 0.5424 | 74,51% | 0.500 |
| 2 | 0.3909 | 0.5926 | 82,83% | 0.500 |
| 3 | 0.2655 | 0.6292 | 88,05% | 0.700 |
| 4 | 0.1437 | 0.8899 | 87,07% | 0.500 |
| 5 | 0.1010 | 1.2232 | 86,41% | 0.500 |
| 6 | 0.1015 | 0.9691 | 87,97% | 0.500 |
| 7 | 0.0656 | 1.3304 | 87,59% | 0.500 |
| 8 | 0.0537 | 1.2094 | 85,61% | 0.500 |
| 9 | 0.0403 | 1.1461 | 87,59% | 0.500 |
| 10 | 0.0129 | 1.1599 | 87,97% | 0.500 |
| 11 | 0.0164 | 1.0983 | 88,05% | 0.500 |
| 12 | 0.0194 | 1.5091 | 86,01% | 0.500 |
| 13 | 0.0177 | 0.9242 | 87,67% | 0.825 |
| 14 | 0.0183 | 1.4040 | 87,97% | 0.500 |
| 15 | 0.0188 | 1.2353 | 87,97% | 0.500 |
| 16 | 0.0105 | 1.2239 | 87,97% | 0.500 |
| 17 | 0.0076 | 1.1686 | 87,97% | 0.500 |
| 18 | 0.0133 | 1.0961 | 87,67% | 0.500 |
| 19 | 0.0121 | 1.1385 | 87,67% | 0.500 |
| 20 | 0.0045 | 1.2269 | 87,97% | 0.500 |

## So sánh trước / sau bằng Validation

Lựa chọn dưới đây đã được khóa trước khi đánh giá Test. Balanced correct tính trung bình tỷ lệ kết luận đúng của từng lớp; ảnh chưa đủ tin cậy tính là chưa kết luận đúng.

| Mô hình | Balanced correct trước | Sau | Báo nhầm trước | Sau | Recall lỗi trước | Sau | Trọng số được dùng |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HOG + SVM | 86,28% | 90,80% | 2 | 0 | 78,53% | 81,60% | Huấn luyện mới |
| HOG + Random Forest | 76,28% | 79,26% | 0 | 0 | 67,48% | 67,48% | Huấn luyện mới |
| MobileNetV3 Small | 84,00% | 89,26% | 0 | 0 | 75,46% | 78,53% | Huấn luyện mới |

Cấu hình chọn bằng cross-validation theo nhóm trong Train; epoch, ngưỡng và quyết định giữ/thay mô hình chỉ dùng Validation. Test chỉ báo cáo sau khi khóa lựa chọn. Tập Test từng được xem trong phát triển, chưa phải kiểm tra mù độc lập. Điểm mô hình chưa hiệu chuẩn.

Bộ trước: `20261115_112043_335868`. Manifest SHA256: `825c4e5158fcffcb58cb417a0cb5aee4172bbcc48a288685cc491ae7d409c63a`. Không thêm ảnh Validation/Test vào Train; ảnh hãng khác vẫn nằm ngoài phạm vi đã kiểm chứng.

## Đối chiếu toàn luồng Test trước / sau

Các lựa chọn đã khóa bằng Validation. Bảng này chỉ báo cáo kết quả trên cùng tập Test phát triển, không dùng để sửa tiếp tham số.

| Mô hình | Đúng / tổng trước | Sau | Báo nhầm bình thường thành lỗi (trước → sau) | Ảnh lỗi bị kết luận bình thường (trước → sau) | Chưa đủ tin cậy (trước → sau) |
| --- | ---: | ---: | ---: | ---: | ---: |
| HOG + SVM | 91,77% | 94,81% | 2 → 1 | 0 → 7 | 17 → 4 |
| HOG + Random Forest | 77,06% | 78,79% | 0 → 0 | 1 → 3 | 52 → 46 |
| MobileNetV3 Small | 73,16% | 87,88% | 1 → 1 | 33 → 22 | 28 → 5 |

Tỷ lệ kết luận đúng cao hơn không đồng nghĩa mọi loại sai số đều giảm. Một số ảnh trước đây chưa đủ tin cậy nay được kết luận bình thường, trong đó vẫn có ảnh lỗi. Cần xem đồng thời số báo nhầm, số ảnh lỗi bị kết luận bình thường và số ảnh chưa kết luận; chưa dùng kết quả này để bảo đảm chất lượng ngoài thực tế.

## Phạm vi kết luận

Dữ liệu có một mẫu chai Ron 88 và bối cảnh chụp giới hạn. Nguồn không có ID chai vật lý; ảnh gần giống có thể còn xuất hiện ở các khối khác nhau. Tập Test đã được xem trong phát triển, bao gồm việc sửa chênh lệch cách cắt ảnh khi học và sử dụng. Đây là số liệu phát triển, chưa phải kiểm tra mù trên chai hoặc phiên chụp độc lập.

Không suy ra khả năng nhận đúng từ mọi góc chụp, ảnh ngoài nguồn hoặc lỗi bị che khuất từ các chỉ số trên. Cần bộ ảnh thật có nhiều góc và các chai/phiên chụp độc lập để đánh giá tiếp. Điểm mô hình chưa được xác nhận là xác suất đúng thực tế.

Thời gian phân loại trên vùng chai, không gồm đọc tệp hoặc bộ định vị chai. CNN đo trên GPU huấn luyện, mô hình cổ điển trên CPU; web ghi thời gian toàn bộ yêu cầu thực tế trên CPU.

Camera vật lý chưa được kiểm thử. Xem [kế hoạch thu thập dữ liệu](THU_THAP_DU_LIEU.md) và [nguồn dữ liệu](DATASET.md).

Bằng chứng gốc: [cấu hình và chỉ số](models/trained/20261115_120255_246951/report.json), [toàn bộ dự đoán web](models/trained/20261115_120255_246951/end_to_end.json). Các lần chạy cũ được giữ riêng; `latest.json` xác định mô hình đang dùng.
