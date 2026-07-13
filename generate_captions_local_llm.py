import json
import os
from tqdm import tqdm

try:
    import ollama
except ImportError:
    print("Vui lòng cài đặt thư viện ollama trước: pip install ollama")
    exit(1)

# Bạn có thể đổi tên model ở đây thành 'qwen2.5:1.5b' hoặc 'phi3' tuỳ ý
MODEL_NAME = 'llama3.2'

def generate_llm_caption(objects, severity, blocked):
    """
    Sinh caption bằng LLM chạy local (Ollama).
    """
    objects_str = ', '.join(objects) if objects else 'no specific obstacles'
    
    prompt = f"""
    You are an AI generating image captions for an Urban Sidewalk Monitoring dataset.
    Based strictly on the following detected attributes, write ONE concise, natural English sentence describing the scene.
    - Detected objects on sidewalk: {objects_str}
    - Violation severity: {severity}
    - Is pedestrian path completely blocked? {blocked}
    
    Output strictly the single sentence without any intro, outro, or quotes.
    """
    
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'user', 'content': prompt}
        ])
        # Lấy nội dung text và xóa khoảng trắng hoặc dấu nháy thừa
        caption = response['message']['content'].strip().strip('"').strip("'").replace('\n', ' ')
        return caption
    except Exception as e:
        # Nếu ollama chưa chạy hoặc model chưa được pull về, báo lỗi
        return f"Error: {e}"

def main():
    json_path = '/home/nguyenhongquan/study/DL/sidewalk/merged_dataset/train/_annotations_merged.coco.json'
    out_path = '/home/nguyenhongquan/study/DL/sidewalk/merged_dataset/train/_annotations_with_llm_captions.coco.json'
    
    print(f"Đọc file {json_path}...")
    if not os.path.exists(json_path):
        print(f"Không tìm thấy file {json_path}")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    cat_id_to_name = {c['id']: c['name'].replace('_', ' ') for c in data['categories']}
    sidewalk_cat_ids = [c['id'] for c in data['categories'] if c['name'].lower() == 'sidewalk']
    ignore_names = ['sw2', 'sw3', 'sw4', 'sw5', 'sw6', 'My-Second-Project', 'My-Third-Project', 'object']
    ignore_cat_ids = [c['id'] for c in data['categories'] if c['name'] in ignore_names]
    
    img_to_anns = {img['id']: [] for img in data['images']}
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
        
    print(f"Đang sinh Caption bằng LLM Local ({MODEL_NAME})...")
    print("Mẹo: Hãy chắc chắn bạn đã bật Ollama và đã chạy 'ollama pull llama3.2' nhé!")
    
    # Chỉ sinh thử cho 50 ảnh đầu tiên để test tốc độ và chất lượng.
    # Khi nào thấy ưng ý, bạn có thể xóa dòng `[:50]` đi để chạy cho toàn bộ 4.7k ảnh.
    test_limit = 50 
    images_to_process = data['images'][:test_limit]
    
    for img in tqdm(images_to_process):
        anns = img_to_anns.get(img['id'], [])
        
        objects = []
        for a in anns:
            cid = a['category_id']
            if cid not in sidewalk_cat_ids and cid not in ignore_cat_ids:
                objects.append(cat_id_to_name[cid])
                
        severity = img.get('violation_severity', 'low')
        blocked = img.get('pedestrian_blocked', 'no')
        
        caption = generate_llm_caption(objects, severity, blocked)
        img['caption_llm'] = caption
            
    # Ghi lại file json mới
    with open(out_path, 'w') as f:
        json.dump(data, f)
        
    print(f"\nHoàn tất sinh caption cho {test_limit} ảnh test.")
    print(f"Đã lưu kết quả tại: {out_path}")
    print("=> Hãy mở file json ra kiểm tra trường 'caption_llm' của các ảnh đầu tiên nhé!")

if __name__ == '__main__':
    main()
