import os
import shutil
import argparse
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from tqdm import tqdm

def variance_of_laplacian(image_path):
    """
    Tính phương sai của bộ lọc Laplacian (variance of Laplacian).
    Giá trị này càng thấp thì ảnh càng có khả năng bị mờ.
    """
    try:
        with Image.open(image_path) as img:
            # Chuyển đổi sang ảnh xám
            img_gray = img.convert('L')
            
            # Áp dụng bộ lọc Laplacian 3x3
            laplacian_kernel = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1)
            edges = img_gray.filter(laplacian_kernel)
            
            # Tính phương sai
            variance = np.var(np.array(edges))
            return variance
    except Exception as e:
        print(f"Lỗi khi đọc file {image_path}: {e}")
        return -1

def filter_blurry_images(input_dir, output_dir, threshold=100.0, action='copy'):
    """
    Lọc các ảnh không bị mờ (độ sắc nét >= threshold) và đưa vào output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Thư mục đầu vào {input_dir} không tồn tại!")
        return

    # Tạo thư mục đầu ra nếu chưa có
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách ảnh
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = [p for p in input_path.rglob('*') if p.is_file() and p.suffix.lower() in valid_extensions]
    
    if not images:
        print(f"Không tìm thấy ảnh nào trong {input_dir}")
        return

    print(f"Đã tìm thấy {len(images)} ảnh. Bắt đầu đánh giá độ sắc nét...")
    print(f"Ngưỡng độ sắc nét (Threshold) đang dùng: {threshold}")
    
    good_images = []
    blurry_images = []
    
    # Đánh giá từng ảnh
    for img_path in tqdm(images, desc="Đang xử lý"):
        score = variance_of_laplacian(img_path)
        if score >= threshold:
            good_images.append((img_path, score))
        elif score != -1:
            blurry_images.append((img_path, score))

    print(f"\nKết quả:")
    print(f" - Ảnh tốt (>= {threshold}): {len(good_images)} ảnh")
    print(f" - Ảnh mờ (< {threshold}): {len(blurry_images)} ảnh")
    
    if len(good_images) > 0:
        print(f"\nĐang {action} các ảnh tốt sang thư mục '{output_dir}'...")
        for img_path, score in tqdm(good_images, desc=f"Đang {action}"):
            # Giữ nguyên cấu trúc thư mục con nếu có, hoặc chỉ copy tên file
            # Ở đây ta sẽ copy trực tiếp vào thư mục đích
            dest_path = output_path / img_path.name
            
            # Đảm bảo không ghi đè nếu trùng tên file bằng cách thêm prefix nếu cần
            # (Đơn giản nhất là để nó ghi đè nếu cùng tên)
            if action == 'copy':
                shutil.copy2(img_path, dest_path)
            elif action == 'move':
                shutil.move(img_path, dest_path)
                
        print(f"Hoàn tất! {len(good_images)} ảnh đã được lưu vào {output_dir}")
    else:
        print("Không có ảnh nào đạt tiêu chuẩn để di chuyển/copy.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lọc ảnh mờ bằng Variance of Laplacian.")
    parser.add_argument("-i", "--input", default="dataset/val/images", help="Thư mục chứa ảnh gốc")
    parser.add_argument("-o", "--output", default="dataset/val1", help="Thư mục lưu ảnh tốt")
    parser.add_argument("-t", "--threshold", type=float, default=100.0, help="Ngưỡng sắc nét (mặc định: 100.0). Tăng lên nếu muốn lọc khắt khe hơn.")
    parser.add_argument("--action", choices=['copy', 'move'], default='copy', help="Hành động: 'copy' hoặc 'move' (mặc định: copy)")
    
    args = parser.parse_args()
    filter_blurry_images(args.input, args.output, threshold=args.threshold, action=args.action)
