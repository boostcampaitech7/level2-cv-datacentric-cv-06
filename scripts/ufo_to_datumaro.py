import json
from pathlib import Path
from utils import extract_flat_points

def process_all_receipts(data_dir="data", split="val"):
    """
    모든 receipt 데이터셋의 UFO 파일을 처리
    """
    # receipt 디렉토리들 찾기
    receipt_dirs = [
        d for d in Path(data_dir).iterdir() 
        if d.is_dir() and "receipt" in d.name and not d.name.startswith(".")
    ]
    
    results = []
    for receipt_dir in receipt_dirs:
        # UFO json 파일 경로
        ufo_path = receipt_dir / "ufo" / f"{split}.json"
        if not ufo_path.exists():
            print(f"Skipping {receipt_dir.name}: {split}.json not found")
            continue
            
        # 출력 디렉토리 설정
        output_dir = Path("annotations") / receipt_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{split}.json"
        
        # 변환 실행
        try:
            output_file = ufo_to_datumaro(
                str(ufo_path), 
                str(output_path),
                task_name=receipt_dir.name,
                split=split
            )
            results.append((receipt_dir.name, output_file))
            print(f"Processed {receipt_dir.name}")
        except Exception as e:
            print(f"Error processing {receipt_dir.name}: {e}")
    
    return results

def ufo_to_datumaro(ufo_json_path, output_path, task_name="receipt_val", split="val"):
    """
    UFO format을 Datumaro format으로 변환
    """
    # UFO 파일 로드
    with open(ufo_json_path, 'r', encoding='utf-8') as f:
        ufo_data = json.load(f)
    
    # CVAT task 관련 설정
    id_prefix = f"{task_name}/images/{split}/"
    
    # 이미지 키와 points 추출
    image_keys = list(ufo_data["images"].keys())
    image_map = {k: v["words"] for k, v in ufo_data["images"].items() if k in image_keys}
    flat_points = {fname: extract_flat_points(image) for fname, image in image_map.items()}
    
    # Datumaro format으로 변환
    def wrap_vertices(vertices):
        return {
            "id": 0,
            "type": "polygon",
            "attributes": {"occluded": False},
            "group": 0,
            "label_id": 0,
            "points": vertices,
            "z_order": 0,
        }
    
    def wrap_annotations(idx, img_name, annotations):
        return {
            "id": id_prefix + img_name,
            "annotations": annotations,
            "attr": {"frame": idx},
            "point_cloud": {"path": ""},
            "info": {
                "img_w": ufo_data["images"][img_name]["img_w"],
                "img_h": ufo_data["images"][img_name]["img_h"],
            },
        }
    
    annotation = {
        "info": {},
        "categories": {
            "label": {
                "labels": [{"name": "1", "parent": "", "attributes": []}],
                "attributes": ["occluded"],
            },
            "points": {"items": []},
        },
        "items": [
            wrap_annotations(
                idx, 
                img_name, 
                [wrap_vertices(vertices) for vertices in points]
            )
            for idx, (img_name, points) in enumerate(flat_points.items())
        ]
    }
    
    # 결과 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)
    
    return output_path

def main():
    # 모든 receipt 데이터 처리
    results = process_all_receipts()
    
    # 결과 출력
    print("\nConversion Results:")
    for dataset_name, output_file in results:
        print(f"{dataset_name}: {output_file}")

if __name__ == "__main__":
    main()