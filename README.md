# Kế Hoạch Thực Hiện Đồ Án SE365

**Đề tài:** Phát hiện hành vi lấn chiếm vỉa hè qua ảnh, sinh mô tả vi phạm và hỗ trợ hỏi đáp trên ảnh bằng Deep Learning và Vision-Language Models.

## 1. Mục Tiêu Đồ Án

Xây dựng hệ thống tự động phân tích ảnh vỉa hè nhằm:
- Phát hiện các hành vi lấn chiếm vỉa hè.
- Phân loại nhiều loại vi phạm trong cùng một ảnh.
- Sinh mô tả bằng ngôn ngữ tự nhiên về tình trạng vi phạm.
- Trả lời các câu hỏi liên quan đến nội dung ảnh.
- Xây dựng ứng dụng web phục vụ trình diễn kết quả.

**Các loại vi phạm dự kiến:**
1. Đỗ xe máy trái phép.
2. Đỗ ô tô trái phép.
3. Buôn bán trên vỉa hè.
4. Đặt biển quảng cáo.
5. Xây dựng hoặc sửa chữa lấn chiếm.
6. Để rác thải trên vỉa hè.
7. Không vi phạm.

---

## 2. Kiến Trúc Hệ Thống

### Module 1: Data Collection
- **Thu thập ảnh từ:** Tự chụp, Video giao thông, Camera công cộng, Google Street View.

### Module 2: Multi-label Classification
- **Input:** Ảnh vỉa hè.
- **Output:** Danh sách các vi phạm xuất hiện trong ảnh.

### Module 3: Violation Captioning
- **Input:** Ảnh.
- **Output:** Mô tả bằng tiếng Việt.
- *Ví dụ:* "Vỉa hè bị lấn chiếm bởi nhiều xe máy đậu trái phép và một biển quảng cáo lớn."

### Module 4: Visual Question Answering (VQA)
- **Input:** Ảnh, Câu hỏi.
- **Output:** Câu trả lời.
- *Ví dụ:* 
  - Q: Có vi phạm không? -> A: Có.
  - Q: Vi phạm gì? -> A: Đỗ xe máy trái phép.

### Module 5: Web Application
- **Chức năng:** Upload ảnh, Hiển thị kết quả phân loại, Hiển thị mô tả, Hỗ trợ hỏi đáp.

---

## 3. Kế Hoạch Thực Hiện (15 Tuần)

### Giai đoạn 1 (Tuần 1): Khảo sát bài toán
- **Nội dung:**
  - Nghiên cứu các công trình liên quan.
  - Khảo sát các mô hình thị giác máy tính hiện đại & Vision-Language Models.
  - Xác định phạm vi đề tài.
- **Kết quả:** Đặc tả yêu cầu, Danh sách nhãn, Kiến trúc tổng thể.
- **Deliverable:** Proposal, Slide báo cáo tiến độ 1.

### Giai đoạn 2 (Tuần 2-4): Thu thập dữ liệu
- **Nội dung:** Chụp ảnh thực tế, Trích xuất frame từ video, Làm sạch dữ liệu, Loại bỏ ảnh trùng lặp.
- **Mục tiêu:** Tối thiểu 4000 ảnh (3000 ảnh có vi phạm, 1000 ảnh không vi phạm).
- **Deliverable:** Dataset v1.

### Giai đoạn 3 (Tuần 3-5): Gán nhãn dữ liệu
- **Nội dung:** Gán nhãn multi-label, Viết caption, Tạo bộ câu hỏi và câu trả lời.
- **Ví dụ:** 
  - *Labels:* `[motorbike_parking, advertising_board]`
  - *Caption:* "Có nhiều xe máy đậu trên vỉa hè cùng một biển quảng cáo lớn."
  - *Question:* "Có vi phạm không?" -> *Answer:* "Có."
- **Mục tiêu:** 4000 ảnh được gán nhãn đầy đủ.
- **Deliverable:** Dataset v2.

### Giai đoạn 4 (Tuần 5-6): Xây dựng mô hình baseline
- **Mô hình:** ResNet50.
- **Loss:** Binary Cross Entropy.
- **Metric:** Accuracy, Precision, Recall, F1 Score.
- **Deliverable:** Baseline Result.

### Giai đoạn 5 (Tuần 6-8): Nghiên cứu và thử nghiệm mô hình nâng cao
- **Mô hình:** ResNet50, EfficientNet-B3, Swin Transformer.
- **Các thí nghiệm:** 
  - Exp 1: Backbone Comparison.
  - Exp 2: BCE Loss.
  - Exp 3: Weighted BCE Loss.
  - Exp 4: Focal Loss.
- **Deliverable:** Bảng so sánh mô hình.

### Giai đoạn 6 (Tuần 8-9): Nghiên cứu cơ chế Attention
- **Thí nghiệm:** CLS Token, Attention Pooling, Gated Fusion.
- **Mục tiêu:** Tìm chiến lược tổng hợp đặc trưng tốt nhất.
- **Deliverable:** Attention Study Report.

### Giai đoạn 7 (Tuần 9-10): Caption Generation
- **Mô hình:** BLIP, BLIP-2, Qwen2.5-VL.
- **Đánh giá:** BLEU, ROUGE-L.
- **Deliverable:** Captioning Module.

### Giai đoạn 8 (Tuần 10-11): Visual Question Answering
- **Mô hình:** BLIP VQA, Qwen2.5-VL.
- **Các loại câu hỏi:** Có vi phạm không?, Vi phạm gì?, Có xe máy không?, Có biển quảng cáo không?, Vỉa hè có bị lấn chiếm không?
- **Đánh giá:** Accuracy, Exact Match.
- **Deliverable:** VQA Module.

### Giai đoạn 9 (Tuần 11-13): Xây dựng hệ thống phần mềm
- **Frontend:** React.
- **Backend:** FastAPI.
- **Database:** PostgreSQL.
- **Chức năng:** Upload ảnh, Phân tích ảnh, Hiển thị kết quả, Hỏi đáp.
- **Deliverable:** MVP System.

### Giai đoạn 10 (Tuần 13-14): Đánh giá và Ablation Study
- **So sánh:** ResNet vs EfficientNet vs Swin, BCE vs Weighted BCE vs Focal Loss, CLS vs Attention Pooling vs Gated Fusion, BLIP vs BLIP-2 vs Qwen2.5-VL.
- **Deliverable:** Experimental Report.

### Giai đoạn 11 (Tuần 14-15): Hoàn thiện báo cáo
- **Nội dung:** Báo cáo các chương (Giới thiệu, Cơ sở lý thuyết, Dữ liệu, Phương pháp đề xuất, Thực nghiệm, Kết quả, Kết luận và hướng phát triển).
- **Deliverable:** Báo cáo hoàn chỉnh, Slide bảo vệ, Source code, Dataset, Demo video.
