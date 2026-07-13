import json
import os

def main():
    merged_json_path = '/home/nguyenhongquan/study/DL/sidewalk/merged_dataset/train/_annotations.coco.json'
    sw6_json_path = '/home/nguyenhongquan/study/DL/sidewalk/sw6.coco/train/_annotations.coco.json'
    
    print(f"Loading merged dataset annotations...")
    with open(merged_json_path, 'r') as f:
        merged_data = json.load(f)
        
    print(f"Loading sw6 annotations...")
    with open(sw6_json_path, 'r') as f:
        sw6_data = json.load(f)
        
    # Tìm ID của category 'sidewalk' trong merged_dataset
    sidewalk_id = None
    for cat in merged_data['categories']:
        if cat['name'] == 'sidewalk':
            sidewalk_id = cat['id']
            break
            
    if sidewalk_id is None:
        print("Error: Không tìm thấy category 'sidewalk' trong merged dataset.")
        return
        
    # Tạo mapping từ tên file gốc sang image_id trong merged_dataset
    filename_to_merged_img_id = {}
    for img in merged_data['images']:
        filename_to_merged_img_id[img['file_name']] = img['id']
        
    # Tạo mapping từ image_id trong sw6 sang tên file gốc
    sw6_img_id_to_filename = {}
    for img in sw6_data['images']:
        fname = img.get('extra', {}).get('name', img['file_name'])
        sw6_img_id_to_filename[img['id']] = fname
        
    # Tìm ID tiếp theo cho annotation mới để không bị trùng lặp
    next_ann_id = 0
    if merged_data['annotations']:
        next_ann_id = max(ann['id'] for ann in merged_data['annotations']) + 1
        
    # Cập nhật thêm các annotation từ sw6
    added_count = 0
    for ann in sw6_data['annotations']:
        fname = sw6_img_id_to_filename.get(ann['image_id'])
        if not fname:
            continue
            
        if fname in filename_to_merged_img_id:
            merged_img_id = filename_to_merged_img_id[fname]
            
            new_ann = ann.copy()
            new_ann['id'] = next_ann_id
            new_ann['image_id'] = merged_img_id
            new_ann['category_id'] = sidewalk_id # Chuyển về đúng ID của sidewalk trong merged_dataset
            
            merged_data['annotations'].append(new_ann)
            next_ann_id += 1
            added_count += 1
        else:
            print(f"Warning: Ảnh {fname} từ sw6 không có mặt trong merged dataset.")
            
    # Lưu lại file JSON
    with open(merged_json_path, 'w') as f:
        json.dump(merged_data, f, indent=4)
        
    print(f"Thành công! Đã thêm (ghi đè) {added_count} annotations mới về 'sidewalk' vào merged dataset.")
    print(f"Tổng số annotations hiện tại: {len(merged_data['annotations'])}")

if __name__ == '__main__':
    main()
