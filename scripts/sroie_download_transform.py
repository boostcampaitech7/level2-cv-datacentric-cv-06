import os
import json
import sys
import shutil

# 현재 폴더에 다운로드 설정
os.system("kaggle datasets download -d urbikn/sroie-datasetv2 -p . --unzip --force")

# 현재 스크립트의 경로를 기준으로 상대 경로 설정
base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# 폴더 경로 설정
txt_folder_path = os.path.join(base_dir, "../SROIE2019/train/box")
output_dir = os.path.join(base_dir, "../SROIE2019/ufo/train")
output_json_path = os.path.join(output_dir, "train.json")

# UFO 폴더가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

def parse_text_file(file_path):
    """텍스트 파일을 읽고 좌표 및 텍스트 데이터를 파싱하여 반환."""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) >= 9:
                # 좌표와 텍스트를 적절히 파싱하여 리스트에 추가
                coordinates = list(map(int, parts[:8]))  # 좌표 부분
                text = ','.join(parts[8:])               # 텍스트 부분 (텍스트에 ','가 포함될 수 있음)
                data.append(coordinates + [text])
    return data

def transform_data(data, img_name, img_width=1280, img_height=1707, index=0):
    """텍스트 데이터를 기반으로 단일 이미지에 대한 JSON 구조 생성."""
    # 이미지 이름을 지정된 형식으로 변경
    formatted_img_name = f"extractor.srioe.train.{index:04d}.jpg"
    return {
        formatted_img_name: {
            "paragraphs": {},
            "words": {
                f"{i:04}": {
                    "transcription": entry[-1],
                    "points": [[entry[0], entry[1]], [entry[2], entry[3]], [entry[4], entry[5]], [entry[6], entry[7]]]
                }
                for i, entry in enumerate(data, start=1)
            },
            "chars": {},
            "img_w": img_width,
            "img_h": img_height,
            "num_patches": None,
            "tags": [],
            "relations": {},
            "annotation_log": {
                "worker": "worker",
                "timestamp": "2024-11-01",
                "tool_version": "",
                "source": None
            },
            "license_tag": {
                "usability": True,
                "public": False,
                "commercial": True,
                "type": None,
                "holder": "Upstage"
            }
        }
    }

def generate_json_from_folder(txt_folder_path, output_json_path, img_width=1280, img_height=1707):
    """폴더 내 모든 텍스트 파일을 읽어 JSON 데이터 생성 후 저장."""
    all_images_data = {"images": {}}
    
    # 폴더에서 텍스트 파일을 순회하며 JSON 구조 생성
    for index, filename in enumerate(os.listdir(txt_folder_path)):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(txt_folder_path, filename)
            data = parse_text_file(file_path)            # 텍스트 파일에서 데이터 읽기
            image_data = transform_data(data, filename.replace(".txt", ".jpg"), img_width, img_height, index)
            all_images_data["images"].update(image_data)
    
    # JSON 파일로 저장
    with open(output_json_path, 'w') as json_file:
        json.dump(all_images_data, json_file, indent=4)
    print(f"JSON 데이터가 {output_json_path}에 저장되었습니다.")

# JSON 생성 실행
generate_json_from_folder(txt_folder_path, output_json_path)

# 폴더 삭제 함수
def remove_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"{folder_path} 폴더가 삭제되었습니다.")
    else:
        print(f"{folder_path} 폴더가 존재하지 않습니다.")

# 폴더 삭제 실행
# 삭제할 폴더 경로 설정
folders_to_delete = [
    os.path.join(base_dir, "../SROIE2019/layoutlm-base-uncased"),
    os.path.join(base_dir, "../SROIE2019/test"),
    os.path.join(base_dir, "../SROIE2019/train/box"),
    os.path.join(base_dir, "../SROIE2019/train/entities")
]

# 모든 지정된 폴더 삭제 실행
for folder in folders_to_delete:
    remove_folder(folder)

# SROIE2019 폴더 이름을 SROIE_receipt으로 변경
old_folder_path = os.path.join(base_dir, "../SROIE2019")
new_folder_path = os.path.join(base_dir, "../sroie_receipt")

if os.path.exists(old_folder_path):
    os.rename(old_folder_path, new_folder_path)
    print(f"{old_folder_path} 폴더가 {new_folder_path}로 이름이 변경되었습니다.")
else:
    print(f"{old_folder_path} 폴더가 존재하지 않습니다.")

# SROIE_receipt/train/img 폴더의 이름을 img/train으로 변경
old_img_folder_path = os.path.join(base_dir, "../sroie_receipt/train/img")
new_img_folder_path = os.path.join(base_dir, "../sroie_receipt/img/train")

# 원본 및 대상 디렉토리 경로
source_dir = '../sroie_receipt/train/img'
destination_dir = '../sroie_receipt/img/train'

# 대상 디렉토리가 존재하지 않으면 생성
os.makedirs(destination_dir, exist_ok=True)

# 원본 디렉토리의 파일 목록 가져오기
files = os.listdir(source_dir)

# 파일들을 대상 디렉토리로 이동
for file in files:
    source_file = os.path.join(source_dir, file)
    destination_file = os.path.join(destination_dir, file)
    
    # 파일이 실제로 존재하고 파일인 경우 이동
    if os.path.isfile(source_file):
        shutil.move(source_file, destination_file)
        print(f'파일 이동: {source_file} -> {destination_file}')
    else:
        print(f'파일이 아닙니다: {source_file}')
