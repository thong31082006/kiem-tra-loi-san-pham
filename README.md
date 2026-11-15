# Kiểm tra chai nhựa — Nhóm 8

Ứng dụng web phân loại một ảnh chai nhựa bằng mô hình đã huấn luyện. Người dùng chọn ảnh từ máy, chụp camera hoặc chọn ảnh trong thư viện, sau đó xem kết quả và lịch sử.

## Tải về và chạy trên Windows

1. Vào [repo GitHub](https://github.com/thong31082006/kiem-tra-loi-san-pham), chọn **Code → Download ZIP** và giải nén. Hoặc dùng `git clone https://github.com/thong31082006/kiem-tra-loi-san-pham.git`.
2. Cài **Python 3.13 64-bit** từ [python.org](https://www.python.org/downloads/). Khi cài, bật **Add Python to PATH**. Nếu dùng lệnh `python`, bản 3.11–3.13 64-bit cũng được hỗ trợ.
3. Mở thư mục chứa `main.py` và nhấp đúp **setup_windows.bat**. Cần Internet để tải thư viện; lần đầu có thể mất vài phút. Chờ thông báo `Setup complete`.
4. Nhấp đúp **run_app.bat**. Trình duyệt mở tại **http://127.0.0.1:8080**. Giữ cửa sổ chương trình mở trong lúc sử dụng; nhấn **Ctrl+C** để dừng.

Bản GitHub đã kèm ba mô hình chất lượng, bộ phân tích loại lỗi và trọng số tìm chai. **Không cần huấn luyện lại, tài khoản Kaggle hay card NVIDIA để kiểm tra ảnh.** Script cài PyTorch bản CPU và tạo môi trường `.venv` ngay trong thư mục repo. Ảnh người dùng và lịch sử mới được lưu cục bộ trên máy chạy.

Tab **Thư viện ảnh** ban đầu trống vì bản tải về không kèm bộ ảnh huấn luyện 1.190 mẫu. Bạn có thể tải ảnh chai thật từ máy hoặc dùng camera ngay. Muốn dùng thư viện và huấn luyện lại, chạy ba lệnh đầu ở mục **Dữ liệu và huấn luyện lại** để tải dữ liệu, tạo manifest và cắt vùng chai. Các ảnh trong `images/demo/` là hình minh họa cũ, không phải bộ kiểm chứng AI trên ảnh chai thật.

Nếu thiếu Python, hãy cài xong rồi mở lại script. Nếu cài thư viện lỗi, kiểm tra mạng và chạy lại `setup_windows.bat`; nếu cổng 8080 bận, chạy `run_app.bat --port 8081`.

## Chạy bằng F5

1. Mở thư mục repo chứa `main.py` bằng VS Code.
2. Cài extension **Python** và **Python Debugger** của Microsoft.
3. Chạy `setup_windows.bat` nếu chưa có môi trường. Script tạo `.venv` trong thư mục repo và cài các thư viện cần thiết.
4. Trong **Run and Debug**, chọn **Chay web kiem tra chai nhua (F5)** rồi nhấn **F5**.
5. Chương trình mở Chrome tại `http://127.0.0.1:8080`. Giữ terminal chạy trong lúc sử dụng; nhấn **Ctrl+C** để dừng.

Có thể mở `run_app.bat`, hoặc từ thư mục mã nguồn chạy:

```powershell
.\.venv\Scripts\python.exe main.py
```

Nếu cổng 8080 đang được dùng, thêm `--port 8081`. Thêm `--no-browser` để chỉ mở máy chủ và tự truy cập địa chỉ trong trình duyệt.

## Sử dụng

1. Trong tab **Kiểm tra**, chọn một ảnh hoặc bấm **Chụp từ camera**. Có thể kéo ảnh vào vùng xem trước.
2. Chọn mô hình và bấm **Kiểm tra ảnh**.
3. Xem nhãn **Bình thường**, **Có dấu hiệu lỗi** hoặc **Chưa đủ tin cậy**, cùng điểm mô hình và thời gian xử lý.
4. Mở **Lịch sử** để xem lại. Mỗi lần kiểm tra được lưu tự động; có thể xóa từng bản ghi.

Tab **Thư viện ảnh** cho phép lọc Train/Validation/Test và nhãn nguồn. Nhãn dưới ảnh là nhãn dữ liệu; mô hình chỉ nhận nội dung ảnh khi dự đoán.

## Các chức năng hỗ trợ trình bày

- **So sánh cả 3 mô hình:** chọn một ảnh rồi bấm nút này trong tab Kiểm tra. Ba mô hình xử lý cùng nội dung ảnh, hiển thị nhãn, điểm và thời gian riêng; có thể mở chi tiết từng kết quả hoặc tải bảng CSV. Mỗi kết quả thành công được lưu riêng trong Lịch sử. Nếu một mô hình gặp lỗi, hai kết quả còn lại vẫn hiển thị.
- **Mô hình đã phân tích gì?:** sau khi kiểm tra thành công, giao diện hiển thị khung vùng chai thực tế được tự cắt (gồm lề), ảnh vùng chai và hình HOG. HOG được vẽ từ vector OpenCV dùng cho SVM/Random Forest, trung bình các ô chồng lấn và chuẩn hóa độ sáng để minh họa. Đây là đặc trưng gradient, chưa phải bản đồ vị trí lỗi hoặc giải thích nguyên nhân dự đoán. MobileNet dùng ảnh RGB, không dùng HOG.
- **Nhãn tham chiếu:** chỉ hiện khi chọn ảnh từ thư viện. Mô tả thiếu nước, thiếu nắp, nắp lỏng/hỏng, dị vật hoặc nhãn hỏng lấy từ annotation gốc. Nhãn này được tách khỏi dự đoán hai lớp của AI và không được gửi vào API phân loại. Ảnh tải lên/camera không được suy đoán nhãn từ tên tệp.
- **Biểu đồ đánh giá:** tab Mô hình có biểu đồ F1, ma trận nhầm lẫn và giải thích Accuracy/Precision/Recall/F1. Đây là chỉ số Test trước bước từ chối thiếu tin cậy; kết quả toàn luồng web được trình bày riêng.
- **Tải lịch sử CSV:** trong tab Lịch sử, tải tối đa 500 lần kiểm tra gần nhất. Tệp có BOM UTF-8 để mở tiếng Việt trong Excel; thời điểm trong CSV là UTC.

Lần dự đoán đầu có thể gồm thời gian nạp mô hình; không dùng riêng thời gian lần đầu để kết luận mô hình nào nhanh nhất. Điểm từng mô hình chưa hiệu chuẩn nên không tự chọn mô hình có điểm cao nhất làm kết luận.

## Bộ AI tự nhận diện loại lỗi

Ứng dụng có bộ phân tích riêng học 6 nhóm từ nội dung vùng chai: bình thường, thiếu mức nước, thiếu nắp, nắp lỏng/hỏng, dị vật nhìn thấy và nhãn hỏng/lệch/thiếu. Bộ này không đọc tên tệp, nhãn thư viện hoặc annotation khi dự đoán. Ba mô hình chất lượng hiện có tiếp tục phân loại bình thường/có lỗi; phần “Loại lỗi AI dự đoán” là kết quả của bộ phân tích bổ sung, không phải giải thích nội tại của SVM/Random Forest. Nếu hai bộ phân tích khác nhau, giao diện yêu cầu đối chiếu và chụp lại.

Huấn luyện bằng dữ liệu có sẵn:

```powershell
.\.venv\Scripts\python.exe scripts\train_defect_types.py --epochs 24 --batch-size 8
```

Giữ nguyên 729 Train / 230 Validation / 231 Test. Các ứng viên SVM, Random Forest và MobileNet được chọn bằng macro-F1 Validation; ngưỡng gọi tên từng lỗi cũng chọn chỉ bằng Validation, yêu cầu Precision tối thiểu 85%, Recall tối thiểu 20% và ít nhất 3 phát hiện đúng. Quyết định lưu trong `models/defect_types/<run_id>/selection_frozen.json` trước khi báo cáo Test. `latest.json` riêng của bộ phân tích loại lỗi không thay ba bộ phân loại chất lượng.

Lần huấn luyện hiện tại bật **thiếu nước, thiếu nắp, nắp lỏng/hỏng và dị vật**. Nhóm **nhãn hỏng/lệch/thiếu** có 17 ảnh Train và chưa phát hiện đúng trên Validation ở mức đủ tin cậy, nên chưa được bật gọi tên tự động. Nhãn hỏng và thiếu nhãn không thể tách riêng do annotation nguồn gộp chung. Một thử nghiệm One-Class SVM vùng nhãn cũng chưa đạt điều kiện Validation, được giữ trong nhật ký thử nghiệm, không kích hoạt.

**Móp chai, nước đục/đổi màu chưa có dữ liệu gán nhãn và chưa hỗ trợ.** Ảnh chưa xác nhận được lực siết/độ kín thực tế của nắp, lượng nước thiếu theo ml, vị trí lỗi hay nhiều loại lỗi đồng thời. Khi chưa đạt ngưỡng, kết quả là “Chưa xác định được loại lỗi”. Báo cáo theo loại và số mẫu hiển thị trong tab Mô hình; lịch sử và CSV lưu riêng kết quả phân tích bổ sung. Bộ dữ liệu Test đã được tham khảo trong phát triển, chưa phải kiểm chứng mù trên chai/phiên chụp mới.

Camera yêu cầu quyền truy cập trong trình duyệt. Chụp rõ toàn bộ chai, gồm nắp và đáy. Nếu vùng cần kiểm tra bị che, hãy đổi góc và chụp lại.

## Kiểm tra lô, phản hồi và trình chiếu

- **Kiểm tra lô:** mở tab Kiểm tra lô, chọn tối đa 50 ảnh (tổng không quá 150 MB), chọn một mô hình rồi bấm Bắt đầu kiểm tra. Ảnh chạy lần lượt, có tiến độ, số ảnh bình thường/có lỗi/chưa kết luận/không xử lý được. Mỗi ảnh thành công được lưu riêng trong Lịch sử. Đây là số ảnh, chưa phải số chai vật lý.
- **Tạm dừng / Tiếp tục:** tạm dừng đợi ảnh hiện tại hoàn tất; các ảnh đang chờ được giữ để tiếp tục. Thử lại ảnh lỗi chỉ đưa các ảnh lỗi hợp lệ về hàng chờ. Gửi lại cùng vị trí/ảnh/mô hình trong một lô trả về bản ghi đã có, không tạo lịch sử trùng. Khi tải lại trang, kết quả đã xử lý còn trong Lịch sử; danh sách tệp đang chờ phải chọn lại.
- **Xem và tải kết quả:** nút Chi tiết của từng ảnh đưa kết quả sang tab Kiểm tra, có ảnh vùng chai và loại lỗi AI bổ sung. Tải kết quả CSV gồm cả ảnh không xử lý được, phiên mô hình và phản hồi người dùng. Các cột văn bản được xử lý để mở trong Excel dưới dạng văn bản.
- **Đánh giá:** bấm Đánh giá kết quả ở phần kết quả ảnh, hoặc Đánh giá trong lô/Lịch sử. Chọn nội dung là kết luận chất lượng hoặc loại lỗi AI bổ sung; chọn Đúng/Sai/Chưa rõ. Nếu Sai, có thể ghi nhãn sửa và ghi chú. Hai nội dung được lưu riêng; có thể sửa đánh giá sau. Lịch sử có bộ lọc theo đánh giá và CSV ghi kèm phản hồi.
- **Phản hồi không tự huấn luyện:** đánh giá nằm trong bảng riêng, không đổi dự đoán gốc, manifest, split hay trọng số AI. Nhãn do người dùng ghi nhận cần được duyệt và phân chia dữ liệu trước khi dùng trong lần huấn luyện sau. Ghi nhận móp chai/nước đục trong phản hồi không đồng nghĩa AI đã hỗ trợ chúng.
- **Trình chiếu:** bấm Trình chiếu trên thanh đầu trang để phóng lớn ảnh và kết quả, vẫn giữ rõ nhãn tham chiếu cùng phạm vi hỗ trợ. Nút Toàn màn hình dùng khi trình duyệt cho phép; nếu bị giới hạn, chế độ phóng lớn vẫn hoạt động. Bấm Thoát trình chiếu hoặc Esc để quay lại. Khi hộp thoại đang mở, Esc đóng hộp thoại trước.

Kiểm tra luồng giao diện bằng `node --test test_workflow.cjs`; API và dữ liệu bằng `.\.venv\Scripts\python.exe test_web_app.py`. Các bài kiểm tra dùng lịch sử tạm, không sửa lịch sử người dùng. Node.js chỉ cần cho bài kiểm tra giao diện, không cần để chạy ứng dụng.

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
.\.venv\Scripts\python.exe scripts\download_defect_dataset.py
.\.venv\Scripts\python.exe scripts\prepare_training_data.py
.\.venv\Scripts\python.exe scripts\prepare_bottle_crops.py
.\.venv\Scripts\python.exe scripts\train_models.py --epochs 8 --batch-size 8 --device auto
.\.venv\Scripts\python.exe scripts\evaluate_web_pipeline.py
.\.venv\Scripts\python.exe scripts\export_report.py
```

Hoặc bấm **Huấn luyện lại** trong tab **Mô hình** để huấn luyện và đánh giá cả ba mô hình trên dữ liệu đã chuẩn bị. Trạng thái được cập nhật trong khi chạy. `--device auto` dùng CUDA nếu có, hoặc CPU. Epoch tốt nhất và ngưỡng phân loại chọn bằng Validation; Test là bộ đánh giá phát triển, không dùng chọn ngưỡng.

`evaluate_web_pipeline.py` chạy toàn bộ ảnh Test bằng `ModelService` thực tế và ghi `end_to_end.json`, không tạo lịch sử kiểm tra. Báo cáo tính ảnh chưa đủ tin cậy hoặc không định vị được chai là chưa có kết luận đúng, và ghi tỷ lệ ảnh được kết luận (Coverage).

Mỗi lần huấn luyện ghi vào `models/trained/<run_id>/`. Chỉ khi cả ba mô hình hoàn tất, `latest.json` mới được chuyển sang lần chạy mới. Nếu huấn luyện lỗi, bộ mô hình trước đó được giữ để tiếp tục sử dụng. Kết quả thực tế và phạm vi kiểm tra được ghi trong [TEST_LOG.txt](TEST_LOG.txt).

### Huấn luyện có đối chiếu bộ đang dùng

```powershell
.\.venv\Scripts\python.exe scripts\retrain_validated.py --epochs 20 --batch-size 8 --activate
```

Luồng này giữ nguyên manifest và split. SVM/Random Forest chọn tham số bằng 3-fold có phân nhóm trong Train; MobileNet fine-tune bốn khối cuối, giữ thống kê BatchNorm cố định và chọn epoch bằng Validation. Ngưỡng 0,65 để từ chối thiếu tin cậy được giữ nguyên.

Mô hình mới chỉ được chọn khi tỷ lệ kết luận đúng cân bằng hai lớp trên Validation tăng, số báo nhầm không tăng và Recall lỗi không giảm quá 2 điểm phần trăm. Ảnh chưa có kết luận được tính là chưa đúng. Mô hình không đạt giữ trọng số và ngưỡng của bộ cũ. Quyết định được lưu trong `selection_frozen.json` trước khi đánh giá Test; không đưa Validation/Test vào Train và không dùng Test để chọn tham số hay quyết định giữ/thay.

Ứng viên và các lần thử nằm trong `models/retraining_candidates/`. Khi đánh giá toàn bộ luồng web hoàn tất, `--activate` sao chép bộ đã chọn vào `models/trained/` và chuyển `latest.json` một cách nguyên tử. Bộ cũ được giữ để đối chiếu. Bỏ `--activate` nếu chỉ muốn huấn luyện và xem báo cáo. Khả năng trên ảnh hãng khác vẫn cần dữ liệu và đánh giá riêng.

## Kiểm tra chương trình

```powershell
.\.venv\Scripts\python.exe test_web_app.py
.\.venv\Scripts\python.exe test_ml_contract.py
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

Bản GitHub không kèm lịch sử sử dụng hay cơ sở dữ liệu trên máy phát triển; `data/web/` được tạo khi chạy. Giao diện Tkinter trước đây nằm trong `desktop_legacy.py`; ứng dụng chính hiện tại là web.
