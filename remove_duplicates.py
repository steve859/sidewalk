import os
import hashlib
import argparse
from pathlib import Path

def get_file_hash(filepath):
    """Tính mã băm MD5 của một tệp để so sánh nội dung."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            # Đọc theo từng chunk để tối ưu bộ nhớ cho file lớn
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Lỗi khi đọc file {filepath}: {e}")
        return None

def remove_duplicates(directory, dry_run=False):
    """Tìm và lọc các tệp ảnh trùng lặp có nội dung y hệt nhau trong thư mục."""
    image_dir = Path(directory)
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"Thư mục {directory} không tồn tại!")
        return

    seen_hashes = set()
    duplicates = []
    
    # Chỉ định các định dạng file ảnh phổ biến
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    
    print(f"Đang quét thư mục: {directory}...")
    
    for filepath in image_dir.rglob('*'):
        if filepath.is_file() and filepath.suffix.lower() in valid_extensions:
            file_hash = get_file_hash(filepath)
            
            if file_hash:
                if file_hash in seen_hashes:
                    duplicates.append(filepath)
                else:
                    seen_hashes.add(file_hash)
                    
    print(f"Đã quét tổng cộng {len(seen_hashes) + len(duplicates)} ảnh.")
    print(f"Tìm thấy {len(duplicates)} ảnh trùng lặp hoàn toàn.")
    
    if len(duplicates) > 0:
        if dry_run:
            print("\n[Chế độ Dry-run] Các file sau sẽ bị xóa nếu chạy thật:")
            for dup in duplicates[:20]: # In ra tối đa 20 file để tránh quá dài
                print(f" - {dup}")
            if len(duplicates) > 20:
                print(f"   ... và {len(duplicates) - 20} file khác.")
        else:
            print("\nĐang tiến hành xóa các file trùng lặp...")
            deleted_count = 0
            for dup in duplicates:
                try:
                    dup.unlink()
                    print(f"Đã xóa: {dup.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Không thể xóa {dup}: {e}")
            print(f"\nHoàn tất! Đã xóa thành công {deleted_count} file trùng lặp.")
    else:
        print("\nKhông có file trùng lặp nào cần xóa.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script để tìm và xóa các ảnh trùng lặp trong thư mục.")
    parser.add_argument("-d", "--dir", default="sidewalk_images", help="Đường dẫn đến thư mục chứa ảnh (mặc định: sidewalk_images)")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ hiển thị các file sẽ bị xóa mà không xóa thật")
    
    args = parser.parse_args()
    remove_duplicates(args.dir, dry_run=args.dry_run)
