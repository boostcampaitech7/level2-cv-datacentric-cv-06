import os
import json
from PIL import Image

def create_image_json_structure(img_path, img_width, img_height):
    """이미지에 대한 기본 JSON 구조 생성."""
    return {
        img_path: {
            "paragraphs": {},
            "words": {
                "0001": {
                    "points": [],
                    "transcription": ""
                }
            },
            "chars": {},
            "img_w": img_width,
            "img_h": img_height,
            "num_patches": None,
            "tags": [],
            "relations": {},
            "annotation_log": {
                "worker": "worker",
                "timestamp": "2024-06-07",
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

def generate_json_for_images(img_folder_path, output_json_path):
    """이미지 폴더의 모든 이미지를 읽어 기본 정보를 JSON 구조로 저장."""
    all_images_data = {"images": {}}
    
    for filename in os.listdir(img_folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):  # 이미지 파일만 선택
            img_path = os.path.join(img_folder_path, filename)
            
            # 이미지 크기 가져오기
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            # 이미지에 대한 JSON 구조 생성 후 추가
            image_data = create_image_json_structure(filename, img_width, img_height)
            all_images_data["images"].update(image_data)
    
    # JSON 파일로 저장
    with open(output_json_path, 'w') as json_file:
        json.dump(all_images_data, json_file, indent=4)
    print(f"JSON 데이터가 {output_json_path}에 저장되었습니다.")

# 예시 사용
img_folder_path = "data/sorie2019/img/test"                  # 이미지 파일이 있는 폴더 경로
output_json_path = "data/sorie2019/ufo/test.json"   # 저장할 JSON 파일 경로
generate_json_for_images(img_folder_path, output_json_path)
