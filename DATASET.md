# Dữ liệu chai nhựa từ Kaggle

## Bộ chính dùng huấn luyện

[Water Bottle Defect Level Detection Dataset](https://www.kaggle.com/datasets/jihadakbr/water-bottle-defect-level-detection-dataset), người cung cấp **jihadakbr**, phiên bản **1**, tải ngày **05/11/2026**. Giấy phép được người cung cấp khai báo trên Kaggle là **MIT**.

- Thư mục: `data/datasets/bottle_defects/`.
- Đã tải 1.500 ảnh và 1.500 tệp nhãn YOLO, cùng `data.yaml`.
- ZIP nguồn có dung lượng 334.467.310 byte.
- Nguồn, phiên bản, thời điểm tải và SHA256 nằm trong `source.json`.
- Tác giả cung cấp mã tại [water-bottle-quality-inspection](https://github.com/jihadakbr/water-bottle-quality-inspection).

SHA256 của ZIP đã tải:

```text
db30051da47788f0caeede7d9c56100822af834920db9a9ad2bdd5a263730666
```

Nhãn gốc mô tả chai Ron 88, chai nhãn hiệu khác và năm loại lỗi: thiếu mức nước, thiếu nắp, nắp lỏng/hỏng, dị vật và nhãn hỏng. Ứng dụng hiện tại gộp thành hai lớp:

- `normal`: ảnh thuộc nhóm chụp `good`, có annotation chai Ron 88 và không có annotation lỗi.
- `defect`: ảnh có ít nhất một annotation lỗi loại 2..6.

100 ảnh chai nhãn hiệu khác được loại khỏi bài toán chất lượng hai lớp. Tên tệp được dùng nhận diện nhóm chụp khi chia dữ liệu; phần huấn luyện và dự đoán chỉ lấy đặc trưng từ pixel ảnh.

## Manifest dùng chung cho ba mô hình

Sau kiểm tra nguồn và chia nhóm có 1.193 ảnh gốc 1280 × 720. Bộ tìm chai tự động từ chối ba ảnh Train/Validation; manifest cuối `data/training/manifest.json` có **1.190 ảnh**:

| Tập | Bình thường | Có lỗi | Tổng |
| --- | ---: | ---: | ---: |
| Train | 374 | 355 | 729 |
| Validation | 67 | 163 | 230 |
| Test | 68 | 163 | 231 |
| Tổng | 509 | 681 | 1.190 |

Dùng seed 42. Các ảnh được gom theo loại tình huống chụp, ngày chụp và khối 40 chỉ số ảnh liên tiếp; mỗi khối chỉ thuộc một split. Bỏ ba ảnh ở mỗi phía ranh giới khối, tổng cộng 207 ảnh, để giảm việc ảnh liên tiếp nằm sát nhau giữa các tập. Không phát hiện bản sao giống hệt theo SHA256 hoặc xung đột nhãn.

Nguồn không cung cấp ID chai vật lý. Cách chia khối này chưa chứng minh rằng Test gồm các chai hoặc phiên chụp độc lập, và vẫn có thể còn ảnh gần giống. Kết quả chỉ phản ánh mẫu Ron 88, camera và ánh sáng của bộ nguồn; cần đánh giá thêm để kết luận cho chai khác hoặc mọi góc chụp.

Các bản ghi giữ ảnh gốc, nhãn, nhóm, split, SHA256 và annotation. `scripts/prepare_bottle_crops.py` dùng SSDLite320 MobileNetV3 Large COCO_V1 để tự định vị chai chỉ từ pixel, với ngưỡng 0,30 và lề 8%. Vùng cắt được lưu PNG không mất dữ liệu tại `data/training/crops/ssdlite_v1/`; `training_path` trỏ tới vùng này. Huấn luyện và web dùng cùng cách cắt, không sử dụng hộp bao nguồn ở bước phân loại cuối. Ba ảnh bị từ chối, vị trí tự tìm và lý do được lưu trong `crop_preprocessing`/`auto_detection` của manifest.

Ba mô hình dùng cùng manifest. Trong mỗi lần chạy, Validation chọn epoch/ngưỡng trước khi tính chỉ số Test. Tuy nhiên, tập Test hiện tại đã được tham khảo trong các vòng phát triển và điều chỉnh vùng đầu vào; báo cáo cuối trên tập này là đánh giá phát triển, chưa phải kiểm tra mù độc lập. Cần giữ riêng ảnh chai/phiên chụp mới để kiểm chứng tiếp.

## Tải và chuẩn bị lại

Từ thư mục mã nguồn, dùng môi trường `.venv-ml`:

```powershell
..\.venv-ml\Scripts\python.exe scripts\download_defect_dataset.py
..\.venv-ml\Scripts\python.exe scripts\prepare_training_data.py --seed 42 --block-size 40 --guard 3
..\.venv-ml\Scripts\python.exe scripts\prepare_bottle_crops.py
```

Script tải dùng API công khai Kaggle, giữ ảnh/nhãn/cấu hình, ghi metadata và xóa ZIP trung gian sau khi hoàn tất. Chạy `scripts/train_models.py` để tạo mô hình; tải dữ liệu riêng không tạo ra mô hình đã huấn luyện.

## Bộ phụ đã tải trước đó

[Industrial Inspection System](https://www.kaggle.com/datasets/dhiaalhemdani/industrial-inspection-system), người cung cấp **dhiaalhemdani**, phiên bản **10**, giấy phép khai báo **MIT**, được giữ tại `data/datasets/plastic_bottles/`.

Bộ phụ có 119 JPEG chai PET thật: 95 ảnh Train và 24 ảnh Val, kích thước 3024 × 3024; đã giải mã đủ bằng OpenCV. Giữ 479 tệp ảnh, annotation và cấu hình, khoảng 127,4 MiB. Bốn nhãn thành phần là `bottle`, `cap`, `label`, `liquid`; không được coi trực tiếp là nhãn chất lượng bình thường/có lỗi. Bộ này hiện không nằm trong manifest huấn luyện ba mô hình hoặc thư viện Test của web.

Metadata và danh sách tệp nằm trong `data/datasets/plastic_bottles/source.json`. SHA256 ZIP nguồn:

```text
d2f580abf8b210a0b8a0b23c4de1edda974abb95e96b1c087631e7e812ab1f93
```

Có thể tải lại bộ phụ bằng `scripts/download_dataset.py`. Các trọng số mô hình của tác giả không được đưa vào ứng dụng; chỉ số trong tài liệu nguồn không phải kết quả của chương trình này.
