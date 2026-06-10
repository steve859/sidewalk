import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2

def get_transforms(aug_level='none', image_size=224):
    """
    aug_level: 'none' (Baseline) hoặc 'basic' (Flip, Rotate, Color)
    """
    if aug_level == 'none':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    elif aug_level == 'basic':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        raise ValueError(f"Unknown augmentation level: {aug_level}")

def get_mixup_cutmix(alpha=0.2, num_classes=7, mode='mixup'):
    """
    Trả về module PyTorch transform (v2) dùng cho Advanced Augmentation.
    Lưu ý: Mixup và Cutmix được apply trực tiếp trên toàn bộ batch dữ liệu trong vòng lặp huấn luyện.
    """
    if mode == 'mixup':
        return v2.MixUp(alpha=alpha, num_classes=num_classes)
    elif mode == 'cutmix':
        return v2.CutMix(alpha=alpha, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown advanced augmentation mode: {mode}")
