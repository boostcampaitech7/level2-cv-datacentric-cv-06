import json
import os
import numpy as np
from shapely.geometry import Polygon

def is_valid_box(points, image_width, image_height, min_area=100, min_side_length=5, 
                aspect_ratio_range=(0.02, 15), boundary_threshold=0):
    """박스의 유효성을 검사하는 함수"""
    try:
        poly = Polygon(points)
        
        # 1. 면적 검사
        if poly.area < min_area:
            return False
        
        # 2. 이미지 범위 검사
        for x, y in points:
            if (x < -boundary_threshold or x > image_width + boundary_threshold or 
                y < -boundary_threshold or y > image_height + boundary_threshold):
                return False
        
        # 3. 변의 길이 검사
        for i in range(4):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % 4]
            side_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if side_length < min_side_length:
                return False
        
        # 4. 종횡비 검사
        bounds = poly.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        if height == 0:
            return False
        aspect_ratio = width / height
        if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
            return False
        
        # 5. 볼록성 검사
        if not poly.is_valid or not poly.is_simple:
            return False
        
        return True
    except Exception:
        return False

def filter_empty_transcriptions(input_json_path, output_json_path):
    """빈 transcription 필터링"""
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for image_id, image_data in data['images'].items():
        filtered_words = {
            word_id: word_info 
            for word_id, word_info in image_data['words'].items() 
            if word_info['transcription'] != ''
        }
        image_data['words'] = filtered_words
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    total_boxes = sum(len(img_data['words']) for img_data in data['images'].values())
    print(f"Filtered JSON saved to: {output_json_path}")
    print(f"Total boxes after filtering: {total_boxes}")

def filter_invalid_boxes(input_json_path, output_json_path):
    """비정상적인 박스 필터링"""
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for image_id, image_data in data['images'].items():
        image_width = image_data['img_w']
        image_height = image_data['img_h']
        
        filtered_words = {
            word_id: word_info 
            for word_id, word_info in image_data['words'].items() 
            if is_valid_box(word_info['points'], image_width, image_height)
        }
        image_data['words'] = filtered_words
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    total_boxes = sum(len(img_data['words']) for img_data in data['images'].values())
    print(f"Filtered JSON saved to: {output_json_path}")
    print(f"Total boxes after filtering: {total_boxes}")

def process_datasets(filter_type='empty'):
    """데이터셋 처리 함수
    Args:
        filter_type: 필터링 타입 ('empty', 'invalid', 'both')
    """
    base_path = "../data"
    
    receipt_datasets = [
        item for item in os.listdir(base_path)
        if item.endswith('_receipt') and os.path.isdir(os.path.join(base_path, item))
    ]
    
    for dataset in receipt_datasets:
        print(f"\nProcessing {dataset} dataset...")
        
        ufo_path = os.path.join(base_path, dataset, 'ufo')
        train_json_path = os.path.join(ufo_path, "train.json")
        
        if not os.path.exists(train_json_path):
            print(f"Warning: {train_json_path} does not exist")
            continue
            
        output_dir = os.path.join(ufo_path, 'filtered')
        os.makedirs(output_dir, exist_ok=True)
        
        if filter_type == 'both':
            # 두 필터 순차적으로 적용
            temp_output = os.path.join(output_dir, "temp.json")
            filter_empty_transcriptions(train_json_path, temp_output)
            output_path = os.path.join(output_dir, f"train_filtered_both.json")
            filter_invalid_boxes(temp_output, output_path)
            os.remove(temp_output)
        else:
            output_path = os.path.join(output_dir, f"train_filtered_{filter_type}.json")
            if filter_type == 'empty':
                filter_empty_transcriptions(train_json_path, output_path)
            elif filter_type == 'invalid':
                filter_invalid_boxes(train_json_path, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Filter receipt dataset')
    parser.add_argument('--filter', type=str, choices=['empty', 'invalid', 'both'],
                      default='empty', help='Filter type to apply')
    args = parser.parse_args()
    
    process_datasets(args.filter)