import json
import os

def main():
    json_path = '/home/nguyenhongquan/study/DL/sidewalk/merged_dataset/train/_annotations.coco.json'
    
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Tìm ID của category 'sidewalk'
    sidewalk_cat_id = None
    for cat in data.get('categories', []):
        if cat['name'] == 'sidewalk':
            sidewalk_cat_id = cat['id']
            break
            
    if sidewalk_cat_id is None:
        print("Không tìm thấy category 'sidewalk' trong dataset.")
        return
        
    # Tập hợp các image_id CÓ chứa label 'sidewalk'
    images_with_sidewalk = set()
    for ann in data.get('annotations', []):
        if ann['category_id'] == sidewalk_cat_id:
            images_with_sidewalk.add(ann['image_id'])
            
    # Lọc ra các ảnh KHÔNG có label 'sidewalk'
    images_without_sidewalk = []
    for img in data.get('images', []):
        if img['id'] not in images_with_sidewalk:
            images_without_sidewalk.append(img['file_name'])
            
    # Hiển thị kết quả
    print(f"Tìm thấy {len(images_without_sidewalk)} ảnh KHÔNG CÓ label 'sidewalk' (trên tổng số {len(data['images'])} ảnh).")
    
    # Lưu danh sách vào file text để dễ theo dõi
    output_txt = '/home/nguyenhongquan/study/DL/sidewalk/merged_dataset/images_no_sidewalk.txt'
    with open(output_txt, 'w') as f:
        for fname in images_without_sidewalk:
            f.write(fname + '\n')
            
    print(f"Danh sách đầy đủ đã được lưu tại: {output_txt}")
    
    if images_without_sidewalk:
        print("Dưới đây là một vài tên ảnh ví dụ:")
        for fname in images_without_sidewalk[:10]:
            print(f" - {fname}")

if __name__ == '__main__':
    main()
