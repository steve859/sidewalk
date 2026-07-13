import json
import os

def main():
    base_dir = '/home/nguyenhongquan/study/DL/sidewalk'
    coco_dirs = ['sw2.coco', 'sw3.coco', 'sw4.coco', 'sw5.coco', 'sw6.coco']
    
    out_dir = os.path.join(base_dir, 'merged_dataset/train')
    output_file = os.path.join(out_dir, '_annotations_only_sw2_to_6.coco.json')
    
    merged_categories = []
    merged_images = []
    merged_annotations = []
    
    cat_name_to_id = {}
    next_cat_id = 0
    
    filename_to_img_id = {}
    next_img_id = 0
    next_ann_id = 0
    
    for c_dir in coco_dirs:
        json_path = os.path.join(base_dir, c_dir, 'train/_annotations.coco.json')
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        cat_map = {}
        for cat in data['categories']:
            name = cat['name']
            if name not in cat_name_to_id:
                cat_name_to_id[name] = next_cat_id
                merged_categories.append({
                    'id': next_cat_id,
                    'name': name,
                    'supercategory': cat.get('supercategory', 'none')
                })
                next_cat_id += 1
            cat_map[cat['id']] = cat_name_to_id[name]
            
        img_map = {} 
        for img in data['images']:
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
            
        for ann in data['annotations']:
            new_ann = ann.copy()
            new_ann['id'] = next_ann_id
            new_ann['image_id'] = img_map[ann['image_id']]
            new_ann['category_id'] = cat_map[ann['category_id']]
            merged_annotations.append(new_ann)
            next_ann_id += 1
            
    merged_data = {
        'info': {},
        'licenses': [],
        'categories': merged_categories,
        'images': merged_images,
        'annotations': merged_annotations
    }
    
    with open(output_file, 'w') as f:
        json.dump(merged_data, f)
        
    print(f"Categories: {len(merged_categories)}")
    print(f"Images: {len(merged_images)}")
    print(f"Annotations: {len(merged_annotations)}")

if __name__ == '__main__':
    main()
