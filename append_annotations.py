import json
import os

def main():
    base_file = '/home/nguyenhongquan/study/DL/sidewalk/merged_dataset/train/_annotations_merged.coco.json'
    
    extra_files = [
        '/home/nguyenhongquan/study/DL/sidewalk/train_2_(6604-7858).coco/_annotations.coco.json'
    ]
    
    # 1. Load base file
    with open(base_file, 'r') as f:
        data = json.load(f)
        
    merged_categories = data['categories']
    merged_images = data['images']
    merged_annotations = data['annotations']
    
    # Rebuild indexes for categories
    cat_name_to_id = {cat['name']: cat['id'] for cat in merged_categories}
    next_cat_id = max([cat['id'] for cat in merged_categories], default=-1) + 1
    
    # Rebuild indexes for images
    filename_to_img_id = {img['file_name']: img['id'] for img in merged_images}
    next_img_id = max([img['id'] for img in merged_images], default=-1) + 1
    
    # Rebuild deduplication set for annotations to prevent duplicates
    # A duplicate annotation is same image_id, same category_id, same bbox
    existing_anns = set()
    for ann in merged_annotations:
        bbox_tuple = tuple(round(x, 2) for x in ann['bbox'])
        existing_anns.add((ann['image_id'], ann['category_id'], bbox_tuple))
        
    next_ann_id = max([ann['id'] for ann in merged_annotations], default=-1) + 1
    
    total_added_images = 0
    total_added_annotations = 0

    for json_path in extra_files:
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} does not exist.")
            continue
            
        print(f"Processing {json_path}...")
        with open(json_path, 'r') as f:
            extra_data = json.load(f)
            
        # Map categories
        cat_map = {}
        for cat in extra_data['categories']:
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
            
        # Map images
        img_map = {}
        for img in extra_data['images']:
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
                total_added_images += 1
                
            img_map[old_id] = filename_to_img_id[orig_name]
            
        # Map annotations
        for ann in extra_data['annotations']:
            mapped_img_id = img_map[ann['image_id']]
            mapped_cat_id = cat_map[ann['category_id']]
            bbox_tuple = tuple(round(x, 2) for x in ann['bbox'])
            
            signature = (mapped_img_id, mapped_cat_id, bbox_tuple)
            if signature not in existing_anns:
                new_ann = ann.copy()
                new_ann['id'] = next_ann_id
                new_ann['image_id'] = mapped_img_id
                new_ann['category_id'] = mapped_cat_id
                merged_annotations.append(new_ann)
                existing_anns.add(signature)
                next_ann_id += 1
                total_added_annotations += 1
                
    # Ghi đè lại base file
    data['categories'] = merged_categories
    data['images'] = merged_images
    data['annotations'] = merged_annotations
    
    with open(base_file, 'w') as f:
        json.dump(data, f)
        
    print(f"\nThống kê sau khi gộp thêm:")
    print(f"Tổng số Categories: {len(merged_categories)}")
    print(f"Tổng số Images: {len(merged_images)} (Đã thêm {total_added_images} ảnh mới)")
    print(f"Tổng số Annotations: {len(merged_annotations)} (Đã thêm {total_added_annotations} bbox mới)")
    print(f"File đã được cập nhật: {base_file}")

if __name__ == '__main__':
    main()
