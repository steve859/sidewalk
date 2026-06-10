Nếu bám sát rubric của SE365, mục tiêu không phải là thử thật nhiều model một cách ngẫu nhiên, mà là:

> Có **baseline → cải tiến → so sánh → phân tích**

Mình đề xuất danh sách model như sau để vừa đủ chiều sâu nghiên cứu, vừa khả thi với sinh viên.

# Module 1: Multi-label Classification (Quan trọng nhất)

Đây là module nên dành **60-70% thời gian nghiên cứu**.

## Baseline Models

### 1. ResNet-50

Vai trò:

* Baseline
* Dễ train
* Ít GPU

Thử nghiệm:

```text
ResNet50 + BCE
```

---

### 2. EfficientNet-B3

Vai trò:

* CNN hiện đại hơn
* Thường tốt hơn ResNet trên dataset vừa

Thử nghiệm:

```text
EfficientNet-B3 + BCE
```

---

### 3. ConvNeXt-Tiny

Vai trò:

* CNN thế hệ mới
* So sánh CNN truyền thống và CNN hiện đại

---

## Transformer Models

### 4. Swin Transformer

Vai trò:

* Vision Transformer
* Đúng yêu cầu mô hình DL hiện đại

---

### 5. ViT-Base

Vai trò:

* Baseline Transformer

---

# Các thí nghiệm bắt buộc

## Experiment A - Backbone

| Model           |
| --------------- |
| ResNet50        |
| EfficientNet-B3 |
| ConvNeXt-Tiny   |
| Swin-Tiny       |
| ViT-Base        |

---

## Experiment B - Loss Function

Mỗi backbone tốt nhất sẽ thử:

### BCE

Baseline

### Weighted BCE

Xử lý imbalance

### Focal Loss

Theo đúng gợi ý giảng viên

### Asymmetric Loss (ASL)

Đây là loss rất mạnh cho multi-label classification.

Nếu làm được sẽ rất có điểm nghiên cứu.

---

## Experiment C - Feature Aggregation

Đúng theo rubric.

### CLS Token

Baseline

### Global Average Pooling

CNN baseline

### Attention Pooling

### Gated Attention Pooling

---

## Experiment D - Data Augmentation

### Không augmentation

### Basic augmentation

* Flip
* Rotate

### Advanced augmentation

* Mixup
* CutMix

---

# Module 2: Violation Description (Captioning)

Mục tiêu:

```text
Ảnh -> Mô tả
```

---

## Baseline

### 1. Template-Based Generation

Ví dụ:

```text
motorbike + advertising
```

↓

```text
Có xe máy đậu trên vỉa hè và có biển quảng cáo.
```

Không phải DL nhưng là baseline cực tốt.

---

## Deep Learning Models

### 2. BLIP

Baseline VLM

---

### 3. BLIP-2

Phiên bản mạnh hơn

---

### 4. Qwen2.5-VL

Rất mạnh hiện nay.

Có khả năng mô tả ảnh phức tạp tốt hơn.

---

## Thí nghiệm

| Model      |
| ---------- |
| Template   |
| BLIP       |
| BLIP-2     |
| Qwen2.5-VL |

---

## Metrics

* BLEU
* ROUGE-L
* Human Evaluation

---

# Module 3: Query trên ảnh (VQA)

Mục tiêu:

```text
Image + Question -> Answer
```

---

## Baseline

### 1. Rule-Based VQA

Dựa vào output của Classification.

Ví dụ:

```text
Q: Có vi phạm không?
```

↓

```text
A: Có
```

---

### 2. Retrieval-Based VQA

Ví dụ:

```text
Q: Vi phạm gì?
```

↓

Lấy từ kết quả Classification

↓

Sinh câu trả lời

---

## Deep Learning

### 3. BLIP VQA

---

### 4. LLaVA

---

### 5. Qwen2.5-VL

---

## Thí nghiệm

| Model      |
| ---------- |
| Rule-Based |
| BLIP VQA   |
| LLaVA      |
| Qwen2.5-VL |

---

# Module mở rộng (Nếu muốn lấy điểm cao)

## Object Detection

Thực ra đề tài của bạn rất hợp với Detection.

---

### 1. YOLOv8

Baseline

---

### 2. YOLOv11

Mạnh hơn

---

### 3. RT-DETR

Transformer Detection

---

## Detect

* xe máy
* ô tô
* biển quảng cáo
* rác
* quầy hàng

---

# Kế hoạch tối ưu cho SE365

Nếu mình là nhóm trưởng, mình sẽ **không làm quá nhiều model ở Caption và VQA**, mà tập trung vào Classification.

## Classification (trọng tâm)

* ResNet50
* EfficientNet-B3
* ConvNeXt-Tiny
* Swin-Tiny

Loss:

* BCE
* Weighted BCE
* Focal Loss
* ASL

Attention:

* GAP
* Attention Pooling
* Gated Attention

---

## Caption

* Template
* BLIP
* Qwen2.5-VL

---

## Query

* Rule-Based
* Qwen2.5-VL

Như vậy bạn sẽ có khoảng **15–20 thí nghiệm** để viết báo cáo, đủ chiều sâu nghiên cứu mà vẫn khả thi trong một học kỳ. Điều này phù hợp hơn nhiều so với việc cố fine-tune quá nhiều VLM lớn rồi không có thời gian làm ablation study.
