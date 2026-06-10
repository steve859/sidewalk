import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from src.models.classifier import create_model
from src.losses.losses import FocalLoss, AsymmetricLoss
from src.dataset.augmentations import get_mixup_cutmix

def get_loss_fn(loss_name, device):
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'weighted_bce':
        # Trọng số ví dụ (sẽ cần tính toán từ phân phối của tập dữ liệu thực tế)
        pos_weight = torch.ones([7]) * 2.0 
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    elif loss_name == 'focal':
        return FocalLoss(alpha=1.0, gamma=2.0)
    elif loss_name == 'asl':
        return AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

def train_epoch(model, dataloader, optimizer, criterion, device, advanced_aug=None):
    model.train()
    total_loss = 0
    # Dummy loop cho việc minh họa pipeline
    # for images, targets in tqdm(dataloader, desc="Training"):
    #     images, targets = images.to(device), targets.to(device)
    #     
    #     if advanced_aug:
    #         # Apply Mixup or Cutmix on batch level
    #         images, targets = advanced_aug(images, targets)
    #         
    #     optimizer.zero_grad()
    #     outputs = model(images)
    #     loss = criterion(outputs, targets)
    #     loss.backward()
    #     optimizer.step()
    #     
    #     total_loss += loss.item()
        
    return total_loss / max(1, len(dataloader) if dataloader else 1)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Khởi tạo Model theo từng experiment
    # model_names hỗ trợ từ timm: 'resnet50', 'efficientnet_b3', 'convnext_tiny', 'swin_tiny_patch4_window7_224', 'vit_base_patch16_224'
    model = create_model(model_name=args.model_name, aggregation=args.aggregation, num_classes=7)
    model.to(device)
    
    # 2. Loss Function
    criterion = get_loss_fn(args.loss, device)
    
    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 4. Advanced Augmentation (Experiment D)
    advanced_aug = None
    if args.advanced_aug in ['mixup', 'cutmix']:
        advanced_aug = get_mixup_cutmix(mode=args.advanced_aug, num_classes=7)
        
    print(f"\n--- Bắt đầu thử nghiệm Classification ---")
    print(f"Device: {device}")
    print(f"Backbone Model (Exp A): {args.model_name}")
    print(f"Loss Function (Exp B): {args.loss}")
    print(f"Aggregation (Exp C): {args.aggregation}")
    print(f"Advanced Augmentation (Exp D): {args.advanced_aug}")
    
    # Huấn luyện mô hình (chờ có dữ liệu thực tế ở Dataset module)
    print("Sẵn sàng huấn luyện khi có dữ liệu!\n")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experiment A
    parser.add_argument('--model_name', type=str, default='resnet50', help='Tên mô hình timm (resnet50, efficientnet_b3, swin_tiny_patch4_window7_224,...)')
    # Experiment C
    parser.add_argument('--aggregation', type=str, default='gap', choices=['gap', 'attention', 'gated_attention', 'cls_token'])
    # Experiment B
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'weighted_bce', 'focal', 'asl'])
    # Experiment D
    parser.add_argument('--advanced_aug', type=str, default='none', choices=['none', 'mixup', 'cutmix'])
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    main(args)
