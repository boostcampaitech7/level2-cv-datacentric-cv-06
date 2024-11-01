import os
import json

def parse_text_file(file_path):
    """텍스트 파일을 읽고 좌표 및 텍스트 데이터를 파싱하여 반환."""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 9:
                # 좌표와 텍스트를 적절히 파싱하여 리스트에 추가
                coordinates = list(map(int, parts[:8]))  # 좌표 부분
                text = parts[8]                          # 텍스트 부분
                data.append(coordinates + [text])
    return data

def transform_data(data, img_path, img_width=1280, img_height=1707):
    """텍스트 데이터를 기반으로 단일 이미지에 대한 JSON 구조 생성."""
    return {
        img_path: {
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
    for filename in os.listdir(txt_folder_path):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(txt_folder_path, filename)
            img_name = filename.replace(".txt", ".jpg")  # 이미지 파일명 유추
            data = parse_text_file(file_path)            # 텍스트 파일에서 데이터 읽기
            image_data = transform_data(data, img_name, img_width, img_height)
            all_images_data["images"].update(image_data)
    
    # JSON 파일로 저장
    with open(output_json_path, 'w') as json_file:
        json.dump(all_images_data, json_file, indent=4)
    print(f"JSON 데이터가 {output_json_path}에 저장되었습니다.")

# 예시 사용
txt_folder_path = "../SROIE2019/txt/train"   # 텍스트 파일들이 있는 폴더 경로 입력
output_json_path = "../SROIE2019/train.json"     # 출력할 JSON 파일 경로
generate_json_from_folder(txt_folder_path, output_json_path)
