import json
from pathlib import Path
from utils import read_json, datum_aro_2_ufo_reduced
from PIL import Image

def add_image_info(datumaro_data, image_dir):
    """datumaro 데이터에 이미지 크기 정보를 추가"""
    image_dir = Path(image_dir)
    
    for item in datumaro_data['items']:
        # info 필드 초기화
        item['info'] = {'img_w': 0, 'img_h': 0}  # 기본값 설정
        
        # 이미지 경로 추출 및 .jpg 확장자 추가
        img_path = item['id'].split('/')[-1]
        if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = f"{img_path}.jpg"
            # item의 id도 업데이트
            item['id'] = '/'.join(item['id'].split('/')[:-1] + [img_path])
            
        # 이미지 파일 찾기
        image_file = next(image_dir.rglob(img_path), None)
        
        if image_file and image_file.is_file():
            try:
                with Image.open(image_file) as img:
                    width, height = img.size
                    item['info']['img_w'] = width
                    item['info']['img_h'] = height
            except Exception as e:
                print(f"Error reading {image_file}: {e}")
    
    return datumaro_data

def main():
    # 입력 디렉토리에서 모든 json 파일 처리
    input_dir = Path("annotations")
    output_base_dir = Path("ufo_reduced")
    image_dir = Path("../data")  # 이미지가 있는 디렉토리
    
    for json_file in input_dir.rglob("*.json"):
        try:
            # Datumaro 형식 읽기
            datumaro_in = read_json(json_file)
            
            # 이미지 크기 정보 추가
            print(f"Processing {json_file}")  # 진행 상황 출력
            datumaro_in = add_image_info(datumaro_in, image_dir)
            
            # UFO 형식으로 변환
            ufo_out = datum_aro_2_ufo_reduced(datumaro_in)
            
            # 출력 경로 설정 및 저장
            relative_path = json_file.relative_to(input_dir)
            output_path = output_base_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(ufo_out, f, indent=2, ensure_ascii=False)
                
            print(f"Converted {json_file} -> {output_path}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            import traceback
            print(traceback.format_exc())  # 상세 에러 메시지 출력

if __name__ == "__main__":
    main()