import json
import os

def filter_empty_transcriptions(input_json_path, output_json_path):
    # JSON 파일 읽기
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 각 이미지에 대해 처리
    for image_id, image_data in data['images'].items():
        # words 딕셔너리에서 빈 transcription 필터링
        filtered_words = {}
        for word_id, word_info in image_data['words'].items():
            if word_info['transcription'] != '':
                filtered_words[word_id] = word_info
        
        # 필터링된 words로 교체
        image_data['words'] = filtered_words
    
    # 결과 저장
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # 통계 출력
    total_boxes = sum(len(img_data['words']) for img_data in data['images'].values())
    print(f"Filtered JSON saved to: {output_json_path}")
    print(f"Total boxes after filtering: {total_boxes}")

def process_all_datasets():
    # 데이터 기본 경로
    base_path = "../data"
    
    # receipt로 끝나는 모든 폴더 찾기
    receipt_datasets = []
    for item in os.listdir(base_path):
        if item.endswith('_receipt') and os.path.isdir(os.path.join(base_path, item)):
            receipt_datasets.append(item)
    
    # 각 데이터셋 처리
    for dataset in receipt_datasets:
        print(f"\nProcessing {dataset} dataset...")
        
        # UFO 폴더 내의 train.json 파일 처리
        ufo_path = os.path.join(base_path, dataset, 'ufo')
        train_json_path = os.path.join(ufo_path, "train.json")
        
        # 출력 디렉토리 생성
        output_dir = os.path.join(ufo_path, 'filtered')
        os.makedirs(output_dir, exist_ok=True)
        
        # train.json 파일만 처리
        if os.path.exists(train_json_path):
            output_path = os.path.join(output_dir, "train_filtered.json")
            print(f"\nProcessing {dataset} - train dataset...")
            filter_empty_transcriptions(train_json_path, output_path)
        else:
            print(f"Warning: {train_json_path} does not exist")

if __name__ == "__main__":
    process_all_datasets()