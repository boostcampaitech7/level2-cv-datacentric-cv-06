import json
import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

def split_train_valid():
    # json 파일들의 경로를 찾습니다
    json_pattern = "/data/ephemeral/home/Dong_Yeong/level2-cv-datacentric-cv-06/code/data/*_receipt/ufo/train.json"
    json_files = glob(json_pattern)

    # 각 json 파일에 대해 처리를 수행합니다
    for json_file in json_files:
        # 파일 경로에서 디렉토리와 파일명을 분리
        json_dir = os.path.dirname(json_file)
        
        # 출력할 파일 경로 설정
        write_train_json = os.path.join(json_dir, 'train.json')
        write_val_json = os.path.join(json_dir, 'val.json')
        
        # json 파일 읽기
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 이미지 리스트 생성 및 분할
        images = list(data['images'].keys())
        train, val = train_test_split(images, train_size=0.8, shuffle=True)
        
        # train과 validation 데이터 생성
        train_images = {img_id: data['images'][img_id] for img_id in train}
        train_anns = {'images': train_images}
        val_images = {img_id: data['images'][img_id] for img_id in val}
        val_anns = {'images': val_images}
        
        # 파일 저장
        with open(write_train_json, 'w') as f:
            json.dump(train_anns, f)
        
        with open(write_val_json, 'w') as f:
            json.dump(val_anns, f)
        
        print(f"처리 완료: {json_file}")
        print(f"Train 이미지 수: {len(train)}, Validation 이미지 수: {len(val)}")

if __name__ == "__main__":
    split_train_valid() 