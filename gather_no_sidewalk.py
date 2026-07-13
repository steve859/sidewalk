import os
import shutil

def main():
    base_dir = '/home/nguyenhongquan/study/DL/sidewalk/merged_dataset'
    txt_file = os.path.join(base_dir, 'images_no_sidewalk.txt')
    src_img_dir = os.path.join(base_dir, 'train/images')
    dst_img_dir = os.path.join(base_dir, 'images_no_sidewalk')
    
    # Tạo folder mới
    os.makedirs(dst_img_dir, exist_ok=True)
    
    # Đọc danh sách ảnh
    with open(txt_file, 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]
        
    copied = 0
    not_found = 0
    for fname in filenames:
        src = os.path.join(src_img_dir, fname)
        dst = os.path.join(dst_img_dir, fname)
        
        if os.path.exists(src):
            shutil.copy2(src, dst) # Dùng copy để không làm hỏng thư mục dataset gốc
            copied += 1
        else:
            print(f"Warning: {src} not found.")
            not_found += 1
            
    print(f"Đã copy thành công {copied} ảnh vào {dst_img_dir}")
    if not_found > 0:
        print(f"Có {not_found} ảnh không tìm thấy.")

if __name__ == '__main__':
    main()
