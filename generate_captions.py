import json
import os
import time
from tqdm import tqdm

# ==============================================================================
# HƯỚNG DẪN DÙNG LLM API (GEMINI - MIỄN PHÍ)
# 1. Cài đặt thư viện: pip install -q -U google-generativeai
# 2. Lấy API Key miễn phí tại: https://aistudio.google.com/app/apikey
# 3. Bỏ comment các dòng bên dưới và nhập API Key của bạn vào.
# ==============================================================================

# import google.generativeai as genai
# genai.configure(api_key="NHẬP_API_KEY_GEMINI_CỦA_BẠN_VÀO_ĐÂY")
# # Dùng mô hình gemini-1.5-flash vì nó cực kỳ nhanh và hoàn toàn miễn phí (được 15 RPM / 1 triệu token 1 phút)
# model = genai.GenerativeModel('gemini-1.5-flash')

def generate_rule_based_caption(objects, severity, blocked):
    """
    Sinh caption dựa trên các Template logic (Baseline).
    Rất nhanh và chuẩn xác, nhưng câu văn thiếu tính đa dạng.
    """
    if not objects:
        return "The sidewalk is clear and fully accessible to pedestrians without any obstruction."
        
    # Đếm số lượng từng loại vật thể (vd: 2 parked motorbikes, 1 table)
    obj_counts = {}
    for obj in objects:
        # Xử lý ngữ pháp số nhiều cơ bản
        plural_obj = obj + "s" if not obj.endswith('s') else obj
        obj_counts[plural_obj] = obj_counts.get(plural_obj, 0) + 1
        
    obj_strings = []
    for obj, count in obj_counts.items():
        if count > 1:
            obj_strings.append(f"{count} {obj}")
        else:
            obj_strings.append(f"a {obj[:-1]}") # Bỏ chữ 's' đi nếu số ít
            
    # Nối chuỗi: "2 parked motorbikes, a table and a garbage bag"
    objects_text = ", ".join(obj_strings[:-1]) + (" and " + obj_strings[-1] if len(obj_strings) > 1 else obj_strings[0])
    
    if blocked == 'yes':
        return f"{objects_text.capitalize()} completely block the pedestrian pathway, forcing people to walk on the roadway."
    elif severity == 'high':
        return f"{objects_text.capitalize()} are heavily occupying the sidewalk, creating a severe obstruction for pedestrians."
    elif severity == 'medium':
        return f"{objects_text.capitalize()} are encroaching on the sidewalk, causing a moderate obstruction but pedestrians can still pass."
    else:
        return f"{objects_text.capitalize()} are present on the sidewalk, causing a minor obstruction, but pedestrians can easily walk past."

def generate_llm_caption(objects, severity, blocked):
    """
    Sinh caption sử dụng LLM API. Câu văn sẽ tự nhiên, đa dạng ngữ cảnh hơn.
    """
    objects_str = ', '.join(objects) if objects else 'none'
    prompt = f"""
    You are an AI generating image captions for an Urban Sidewalk Monitoring dataset.
    Based strictly on the following detected attributes, write ONE concise, natural English sentence describing the scene.
    - Detected objects on sidewalk: {objects_str}
    - Violation severity: {severity}
    - Is pedestrian path completely blocked? {blocked}
    
    Do not add hallucinated details. Keep it objective and descriptive like a standard COCO caption.
    """
    
    # response = model.generate_content(prompt)
    # return response.text.strip().replace('\n', ' ')
    
    # (Tạm thời return rỗng nếu chưa bật API)
    return "LLM Caption requires API key."

def main():
    json_path = '/home/nguyenhongquan/study/DL/sidewalk/merged_dataset/train/_annotations_merged.coco.json'
    out_path = '/home/nguyenhongquan/study/DL/sidewalk/merged_dataset/train/_annotations_with_captions.coco.json'
    
    print(f"Reading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    cat_id_to_name = {c['id']: c['name'].replace('_', ' ') for c in data['categories']}
    sidewalk_cat_ids = [c['id'] for c in data['categories'] if c['name'].lower() == 'sidewalk']
    ignore_names = ['sw2', 'sw3', 'sw4', 'sw5', 'sw6', 'My-Second-Project', 'My-Third-Project', 'object']
    ignore_cat_ids = [c['id'] for c in data['categories'] if c['name'] in ignore_names]
    
    img_to_anns = {img['id']: [] for img in data['images']}
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
        
    print("Generating captions...")
    # Bật API thì cân nhắc lấy số lượng ít để test trước (vd: data['images'][:10])
    for img in tqdm(data['images']):
        anns = img_to_anns.get(img['id'], [])
        
        # Lấy danh sách các vật cản (trừ sidewalk và các nhãn meta)
        objects = []
        for a in anns:
            cid = a['category_id']
            if cid not in sidewalk_cat_ids and cid not in ignore_cat_ids:
                objects.append(cat_id_to_name[cid])
                
        severity = img.get('violation_severity', 'low')
        blocked = img.get('pedestrian_blocked', 'no')
        
        # 1. Sinh Caption bằng Rule-based
        caption_rule_based = generate_rule_based_caption(objects, severity, blocked)
        img['caption_rule_based'] = caption_rule_based
        
        # 2. Sinh Caption bằng LLM API 
        # (NẾU DÙNG API THÌ BỎ COMMENT KHỐI TRY-EXCEPT DƯỚI ĐÂY)
        # try:
        #     img['caption_llm'] = generate_llm_caption(objects, severity, blocked)
        #     time.sleep(0.5) # Chờ 0.5s để tránh dính Rate Limit của API Free
        # except Exception as e:
        #     print(f"API Error at img {img['file_name']}: {e}")
        #     img['caption_llm'] = "API Error"
            
    with open(out_path, 'w') as f:
        json.dump(data, f) # Không dùng indent để tránh file bị phình to
        
    print(f"\nSaved dataset with captions to {out_path}")

if __name__ == '__main__':
    main()
