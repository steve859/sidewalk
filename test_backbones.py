import torch
import timm
from src.models.classifier import create_model

def run_experiments():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Timm Version: {timm.__version__}\n")

    # Định nghĩa 5 backbone để thí nghiệm
    backbones = {
        "ResNet50": "resnet50",
        "EfficientNet": "efficientnet_b3",
        "ConvNeXt": "convnext_tiny",
        "Swin Transformer": "swin_tiny_patch4_window7_224",
        "CLIP ViT": "vit_base_patch16_clip_224.openai" # Lưu ý: Cần timm >= 0.6.0
    }

    dummy_input = torch.randn(2, 3, 224, 224) # Batch size = 2, 3 channels, 224x224
    num_classes = 7 # Số lượng class trong dataset của bạn

    for name, timm_name in backbones.items():
        print(f"{'='*50}")
        print(f"Khởi tạo mô hình: {name} (timm: {timm_name})")
        
        # Chọn chiến lược gom đặc trưng (aggregation)
        # Các mạng CNN dùng Global Average Pooling (GAP)
        # Các mạng ViT dùng token [CLS]
        agg_type = 'cls_token' if 'vit' in timm_name.lower() else 'gap'
        
        try:
            # Gọi hàm create_model từ src/models/classifier.py
            model = create_model(
                model_name=timm_name, 
                aggregation=agg_type, 
                num_classes=num_classes
            )
            
            # Đếm số lượng tham số (trainable parameters)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f" -> Số lượng tham số (Parameters): {total_params:,}")
            
            # Đưa dummy data qua mô hình để test kiến trúc
            model.eval() # Chuyển sang chế độ evaluation để tránh lỗi BatchNorm
            with torch.no_grad():
                outputs = model(dummy_input)
                
            print(f" -> Kích thước Output (Logits shape): {list(outputs.shape)} (Kỳ vọng: [2, {num_classes}])")
            print(f" -> Trạng thái: THÀNH CÔNG ✅")
            
        except Exception as e:
            print(f" -> Trạng thái: THẤT BẠI ❌")
            print(f" -> Lỗi chi tiết: {e}")
            
            # Gợi ý fallback nếu bị lỗi tên CLIP ViT ở các bản timm khác nhau
            if name == "CLIP ViT":
                print("    * Gợi ý: Nếu báo lỗi không tìm thấy model CLIP, hãy thử đổi thành 'vit_base_patch16_224' hoặc cập nhật timm: pip install timm --upgrade")
            
    print(f"{'='*50}\nDone!")

if __name__ == "__main__":
    run_experiments()
