import torch
import torch.nn as nn
import timm
from src.models.aggregations import GlobalAveragePooling, AttentionPooling, GatedAttentionPooling

class SidewalkClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=7, pretrained=True, aggregation='gap'):
        super().__init__()
        self.model_name = model_name
        self.aggregation_type = aggregation
        
        # Load model từ timm, bỏ đi lớp classifier cuối (num_classes=0) và không dùng global_pool mặc định
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        
        # Xác định số chiều (C) của feature map sinh ra từ backbone
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            
        if features.dim() == 4:
            in_features = features.shape[1] # (B, C, H, W)
        elif features.dim() == 3:
            in_features = features.shape[2] # (B, N, C)
        else:
            in_features = features.shape[1] # fallback
            
        # Feature Aggregation Module
        if aggregation == 'gap':
            self.aggregation = GlobalAveragePooling()
        elif aggregation == 'attention':
            self.aggregation = AttentionPooling(in_features)
        elif aggregation == 'gated_attention':
            self.aggregation = GatedAttentionPooling(in_features)
        elif aggregation == 'cls_token':
            # Đối với mô hình hỗ trợ CLS token (như ViT), chúng ta xử lý ở hàm forward
            self.aggregation = 'cls_token'
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
            
        # Lớp phân loại cuối cùng
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        
        if self.aggregation_type == 'cls_token':
            if features.dim() == 3:
                # Lấy token đầu tiên (giả sử đó là CLS token cho ViT)
                pooled = features[:, 0]
            else:
                # Nếu model không phải ViT mà vẫn truyền 'cls_token', ta fallback về GAP hoặc quăng lỗi
                # raise ValueError("CLS token aggregation is only supported for transformers returning 3D tensors.")
                pooled = features.mean(dim=[2, 3]) if features.dim() == 4 else features.mean(dim=1)
        else:
            pooled = self.aggregation(features)
            
        logits = self.head(pooled)
        return logits

def create_model(model_name, aggregation='gap', num_classes=7):
    """
    Hàm hỗ trợ khởi tạo model nhanh
    - model_name: 'resnet50', 'efficientnet_b3', 'convnext_tiny', 'swin_tiny_patch4_window7_224', 'vit_base_patch16_224'
    """
    return SidewalkClassifier(model_name=model_name, num_classes=num_classes, pretrained=True, aggregation=aggregation)
