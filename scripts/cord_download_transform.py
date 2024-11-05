import os
import json
from datasets import load_dataset

ds = load_dataset("naver-clova-ix/cord-v2")

def convert_cord_to_ufo(cord_data, split):
    ufo_data = {
        "images": {}
    }
    
    for idx, image_info in enumerate(cord_data['image']):
        img_filename = f'extractor.cord.{split}.{idx:04d}.jpg'
        words_info = {}
        valid_words = []
        
        # valid_line에서 단어 정보 추출
        gt_data = json.loads(cord_data['ground_truth'][idx])
        for line in gt_data['valid_line']:
            for word in line['words']:
                # quad 값 처리
                quad = word['quad']
                points = [
                    [float(quad['x1']), float(quad['y1'])],  # 좌상
                    [float(quad['x2']), float(quad['y2'])],  # 우상
                    [float(quad['x3']), float(quad['y3'])],  # 우하
                    [float(quad['x4']), float(quad['y4'])]   # 좌하
                ]
                
                # 빈 텍스트는 건너뛰기
                if not word['text'].strip():
                    continue
                
                valid_words.append({
                    "transcription": word['text'],
                    "points": points
                })
        
        # words_info 딕셔너리 생성
        for word_idx, word in enumerate(valid_words, 1):
            words_info[str(word_idx).zfill(4)] = word
        
        # 이미지 정보 추가
        image_info_ufo = {
            "words": words_info,
            "paragraphs": {},
            "chars": {},
            "img_w": image_info.shape[1] if hasattr(image_info, 'shape') else 1000,
            "img_h": image_info.shape[0] if hasattr(image_info, 'shape') else 1000,
            "tags": [],
            "relations": {},
            "annotation_log": {
                "worker": "worker",
                "timestamp": "2024-05-30",
                "tool_version": "",
                "source": None
            },
            "license_tag": {
                "usability": True,
                "public": False,
                "commercial": True,
                "type": None,
                "holder": "CORD"
            }
        }
        
        ufo_data["images"][img_filename] = image_info_ufo
    
    return ufo_data
# 디렉토리 생성
base_dir = './cord_receipt'
os.makedirs(os.path.join(base_dir, 'img/train'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'img/test'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'ufo'), exist_ok=True)

# train 데이터 처리
train_data = convert_cord_to_ufo(ds['train'], 'train')
for idx, img in enumerate(ds['train']['image']):
    img_filename = f'extractor.cord.train.{idx:04d}.jpg'
    img_path = os.path.join(base_dir, 'img/train', img_filename)
    img.save(img_path)
    train_data['images'][img_filename]['img_path'] = f'train/{img_filename}'
    

# validation 데이터 처리
val_data = convert_cord_to_ufo(ds['validation'], 'val')
for idx, img in enumerate(ds['validation']['image']):
    img_filename = f'extractor.cord.val.{idx:04d}.jpg'
    img_path = os.path.join(base_dir, 'img/train', img_filename)
    img.save(img_path)
    val_data['images'][img_filename]['img_path'] = f'train/{img_filename}'
    

# test 데이터 처리
test_data = convert_cord_to_ufo(ds['test'], 'test')
for idx, img in enumerate(ds['test']['image']):
    img_filename = f'extractor.cord.test.{idx:04d}.jpg'
    img_path = os.path.join(base_dir, 'img/test', img_filename)
    img.save(img_path)
    test_data['images'][img_filename]['img_path'] = f'test/{img_filename}'

# UFO 데이터셋을 JSON 파일로 저장
with open(os.path.join(base_dir, 'ufo/train.json'), 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(os.path.join(base_dir, 'ufo/val.json'), 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

with open(os.path.join(base_dir, 'ufo/test.json'), 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("이미지와 UFO 데이터셋이 성공적으로 저장되었습니다.")