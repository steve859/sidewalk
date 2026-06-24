import argparse
import os
import csv
from datetime import datetime
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
    
    # ---------------------------------------------------------
    # KHỞI TẠO THƯ MỤC LƯU CHECKPOINT (Giúp bảo vệ trọng số)
    # ---------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Khởi tạo Model theo từng experiment
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
    print(f"Thư mục lưu an toàn (Checkpoint Dir): {args.save_dir}")
    print(f"Backbone: {args.model_name} | Loss: {args.loss} | Aggregation: {args.aggregation}")
    
    # ---------------------------------------------------------
    # HUẤN LUYỆN VÀ TỰ ĐỘNG LƯU TRỌNG SỐ (SAVE CHECKPOINTS)
    # ---------------------------------------------------------
    best_loss = float('inf')
    # dummy_dataloader = [] # Sẽ thay bằng DataLoader thật ở Phase sau
    
    print("\n[Mô phỏng vòng lặp huấn luyện...]")
    for epoch in range(args.epochs):
        # train_loss = train_epoch(model, dummy_dataloader, optimizer, criterion, device, advanced_aug)
        
        # Giả lập train_loss giảm dần để test logic lưu file
        train_loss = 1.0 / (epoch + 1)
        
        # 1. LƯU MÔ HÌNH SAU MỖI EPOCH (Để phòng hờ rớt mạng có thể resume lại)
        last_model_path = os.path.join(args.save_dir, 'last_model.pth')
        torch.save(model.state_dict(), last_model_path)
        
        # 2. LƯU MÔ HÌNH TỐT NHẤT (Dùng để mang đi test thực tế trên Web App)
        if train_loss < best_loss:
            best_loss = train_loss
            best_model_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1}/{args.epochs}: Loss cải thiện ({best_loss:.4f}) -> Đã lưu '{best_model_path}'")
        else:
            print(f"Epoch {epoch+1}/{args.epochs}: Loss ({train_loss:.4f}) -> Đã cập nhật '{last_model_path}'")
            
    print(f"\nHoàn tất huấn luyện. Trọng số của bạn đã được bảo vệ an toàn tại: {args.save_dir}")
    
    # ---------------------------------------------------------
    # LƯU NHẬT KÝ THỬ NGHIỆM (LOGGING) VÀO CSV
    # ---------------------------------------------------------
    log_file = os.path.join(args.save_dir, 'experiments_log.csv')
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Ghi Header nếu file chưa tồn tại
        if not file_exists:
            writer.writerow(['Thời_gian', 'Ghi_chú_mô_tả', 'Model', 'Loss_Fn', 'Aggregation', 'Augmentation', 'Best_Loss'])
        
        # Ghi kết quả của lần chạy này
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            current_time, 
            args.note, 
            args.model_name, 
            args.loss, 
            args.aggregation, 
            args.advanced_aug, 
            f"{best_loss:.4f}"
        ])
    print(f"-> Đã ghi log kết quả thực nghiệm vào: {log_file}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # =========================================================
    # QUAN TRỌNG: Thêm tham số đường dẫn lưu file (--save_dir)
    # =========================================================
    parser.add_argument('--save_dir', type=str, default='./checkpoints', 
                        help='Thư mục lưu trọng số. Khi chạy trên Colab, bạn hãy truyền path của Google Drive vào đây.')
    
    # Các tham số của Experiments
    parser.add_argument('--note', type=str, default='Không có ghi chú', 
                        help='Ghi chú mô tả cho lần chạy này (vd: "Test ResNet với Focal Loss để khắc phục imbalance")')
    
    parser.add_argument('--model_name', type=str, default='resnet50', help='Tên mô hình timm')
    parser.add_argument('--aggregation', type=str, default='gap', choices=['gap', 'attention', 'gated_attention', 'cls_token'])
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'weighted_bce', 'focal', 'asl'])
    parser.add_argument('--advanced_aug', type=str, default='none', choices=['none', 'mixup', 'cutmix'])
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    main(args)
