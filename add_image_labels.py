import json
import os
import csv

def bbox_intersection(box1, box2):
    # COCO bbox: [x, y, width, height]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to [xmin, ymin, xmax, ymax]
    b1_xmin, b1_ymin, b1_xmax, b1_ymax = x1, y1, x1+w1, y1+h1
    b2_xmin, b2_ymin, b2_xmax, b2_ymax = x2, y2, x2+w2, y2+h2
    
    intersect_xmin = max(b1_xmin, b2_xmin)
    intersect_ymin = max(b1_ymin, b2_ymin)
    intersect_xmax = min(b1_xmax, b2_xmax)
    intersect_ymax = min(b1_ymax, b2_ymax)
    
    if intersect_xmax <= intersect_xmin or intersect_ymax <= intersect_ymin:
        return 0, 0, 0 # area, width, height
    
    return (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin), (intersect_xmax - intersect_xmin), (intersect_ymax - intersect_ymin)

def main():
    json_path = '/home/nguyenhongquan/study/DL/sidewalk/merged_dataset/train/_annotations_merged.coco.json'
    csv_path = '/home/nguyenhongquan/study/DL/sidewalk/merged_dataset/train/image_labels.csv'
    
    print(f"Đọc file {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    sidewalk_cat_ids = [c['id'] for c in data['categories'] if c['name'].lower() == 'sidewalk']
    
    img_to_anns = {img['id']: [] for img in data['images']}
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
        
    stats = {'low': 0, 'medium': 0, 'high': 0, 'blocked_yes': 0, 'blocked_no': 0}
    
    csv_data = []
    
    for img in data['images']:
        anns = img_to_anns.get(img['id'], [])
        
        sidewalk_anns = [a for a in anns if a['category_id'] in sidewalk_cat_ids]
        # Ignore metadata categories like sw2, sw3, sw4, My-Second-Project by checking name
        # We consider actual physical objects as obstacles
        ignore_names = ['sw2', 'sw3', 'sw4', 'sw5', 'sw6', 'My-Second-Project', 'My-Third-Project', 'object']
        ignore_cat_ids = [c['id'] for c in data['categories'] if c['name'] in ignore_names]
        
        obstacle_anns = [a for a in anns if a['category_id'] not in sidewalk_cat_ids and a['category_id'] not in ignore_cat_ids]
        
        severity = 'low'
        blocked = 'no'
        
        if sidewalk_anns and obstacle_anns:
            total_sidewalk_area = sum(a['bbox'][2] * a['bbox'][3] for a in sidewalk_anns)
            
            if total_sidewalk_area > 0:
                total_overlap_area = 0
                max_width_block_ratio = 0
                max_height_block_ratio = 0
                
                for obs in obstacle_anns:
                    obs_overlap_area = 0
                    for sw in sidewalk_anns:
                        area, w, h = bbox_intersection(obs['bbox'], sw['bbox'])
                        obs_overlap_area += area
                        
                        sw_width = sw['bbox'][2]
                        sw_height = sw['bbox'][3]
                        
                        if sw_width > 0:
                            w_ratio = w / sw_width
                            if w_ratio > max_width_block_ratio:
                                max_width_block_ratio = w_ratio
                                
                        if sw_height > 0:
                            h_ratio = h / sw_height
                            if h_ratio > max_height_block_ratio:
                                max_height_block_ratio = h_ratio
                                
                    total_overlap_area += obs_overlap_area
                    
                overlap_ratio = total_overlap_area / total_sidewalk_area
                
                # Logic Blocked:
                # Nghiêm khắc hơn: Chắn > 50% chiều ngang/dọc, hoặc chiếm > 40% diện tích, hoặc có từ 3 vật cản
                if max_width_block_ratio > 0.5 or max_height_block_ratio > 0.5 or overlap_ratio > 0.4 or len(obstacle_anns) >= 3:
                    blocked = 'yes'
                    
                # Logic Severity:
                if blocked == 'yes' or overlap_ratio > 0.3:
                    severity = 'high'
                elif overlap_ratio > 0.05 or len(obstacle_anns) > 0:
                    severity = 'medium'
                else:
                    severity = 'low'
                    
        # Nếu không có sidewalk nhưng có nhiều vật cản, có thể default là medium/high tuỳ diện tích, 
        # nhưng an toàn nhất là đánh giá trên những ảnh có sidewalk. 
        # Tạm thời nếu ko có sidewalk -> low, no
        
        # Ghi nhận vào dictionary của ảnh
        img['violation_severity'] = severity
        img['pedestrian_blocked'] = blocked
        
        stats[severity] += 1
        if blocked == 'yes':
            stats['blocked_yes'] += 1
        else:
            stats['blocked_no'] += 1
            
        csv_data.append({
            'file_name': img['file_name'],
            'violation_severity': severity,
            'pedestrian_blocked': blocked,
            'num_obstacles': len(obstacle_anns)
        })
        
    print(f"Cập nhật json data...")
    with open(json_path, 'w') as f:
        json.dump(data, f)
        
    print(f"Xuất file CSV...")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file_name', 'violation_severity', 'pedestrian_blocked', 'num_obstacles'])
        writer.writeheader()
        writer.writerows(csv_data)
        
    print("\nHoàn tất! Thống kê:")
    print(f" - Mức độ vi phạm (Severity): Low ({stats['low']}), Medium ({stats['medium']}), High ({stats['high']})")
    print(f" - Bị chắn đường (Blocked): Yes ({stats['blocked_yes']}), No ({stats['blocked_no']})")
    print(f"\nFile JSON đã được cập nhật thêm trường 'violation_severity' và 'pedestrian_blocked' cho mỗi ảnh.")
    print(f"File CSV xem nhanh được lưu tại: {csv_path}")

if __name__ == '__main__':
    main()
