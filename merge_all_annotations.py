import json
import os
import glob

def main():
    base_dir = '/home/nguyenhongquan/study/DL/sidewalk/merged_dataset'
    
    # Tìm tất cả các file JSON annotation trong thư mục này
    json_files = glob.glob(os.path.join(base_dir, '**/*.json'), recursive=True)
    
    # Loại bỏ file output dự kiến để tránh loop/đọc nhầm file mình vừa ghi
    output_file = os.path.join(base_dir, 'train', '_annotations_merged.coco.json')
    if output_file in json_files:
        json_files.remove(output_file)
        
    print(f"Tìm thấy {len(json_files)} file annotation:")
    for f in json_files:
        print(f" - {f}")
        
    merged_categories = []
    merged_images = []
    merged_annotations = []
    
    cat_name_to_id = {}
    next_cat_id = 0
    
    filename_to_img_id = {}
    next_img_id = 0
    next_ann_id = 0
    
    info = None
    licenses = []
    
    total_original_images = 0
    total_original_annotations = 0
    
    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        if info is None and 'info' in data:
            info = data['info']
        if not licenses and 'licenses' in data:
            licenses = data['licenses']
            
        # 1. Map categories
        cat_map = {} # old_id -> new_id
        for cat in data['categories']:
            name = cat['name']
            if name not in cat_name_to_id:
                cat_name_to_id[name] = next_cat_id
                merged_categories.append({
                    'id': next_cat_id,
                    'name': name,
                    'supercategory': 'none' if name.startswith('sw') else cat.get('supercategory', 'none')
                })
                next_cat_id += 1
            cat_map[cat['id']] = cat_name_to_id[name]
            
        # 2. Map images
        img_map = {} # old_id -> new_id
        for img in data['images']:
            total_original_images += 1
            old_id = img['id']
            rf_filename = img['file_name']
            orig_name = img.get('extra', {}).get('name', rf_filename)
            
            if orig_name not in filename_to_img_id:
                filename_to_img_id[orig_name] = next_img_id
                new_img = img.copy()
                new_img['id'] = next_img_id
                new_img['file_name'] = orig_name
                merged_images.append(new_img)
                next_img_id += 1
                
            img_map[old_id] = filename_to_img_id[orig_name]
            
        # 3. Map annotations
        for ann in data['annotations']:
            total_original_annotations += 1
            new_ann = ann.copy()
            new_ann['id'] = next_ann_id
            new_ann['image_id'] = img_map[ann['image_id']]
            new_ann['category_id'] = cat_map[ann['category_id']]
            merged_annotations.append(new_ann)
            next_ann_id += 1
            
    merged_data = {
        'info': info or {},
        'licenses': licenses,
        'categories': merged_categories,
        'images': merged_images,
        'annotations': merged_annotations
    }
    
    with open(output_file, 'w') as f:
        json.dump(merged_data, f)
        
    print(f"\nThống kê sau khi merge:")
    print(f"Tổng số Categories: {len(merged_categories)}")
    print(f"Tổng số Images: {len(merged_images)} (từ {total_original_images} ảnh gốc, có thể có trùng lặp file_name)")
    print(f"Tổng số Annotations: {len(merged_annotations)} (từ {total_original_annotations} bbox gốc)")
    print(f"\nFile đã được lưu tại: {output_file}")

if __name__ == '__main__':
    main()
