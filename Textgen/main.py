import os
import json
import argparse

from tqdm.auto import tqdm  # tqdm import 방식 수정
from utils.image_generator import ImageGenerator

def _infer_dir(lang_indicator):
        if lang_indicator == 'zh':
            lang = 'chinese'
        elif lang_indicator == 'ja':
            lang = 'japanese'
        elif lang_indicator == 'th':
            lang = 'thai'
        elif lang_indicator == 'vi':
            lang = 'vietnamese'
        else:
            raise ValueError
        return lang

def generate_dataset(languages, num_images=10, sentences_per_image=64):
    total_progress = tqdm(
        range(len(languages)),
        desc="전체 진행률",
        position=0
    )
    
    for lang_idx in total_progress:
        lang = languages[lang_idx]
        full_lang = _infer_dir(lang)
        root_dir = f'data/{full_lang}_receipt'

        os.makedirs(f'{root_dir}/img', exist_ok=True)
        os.makedirs(f'{root_dir}/ufo', exist_ok=True)
        print(f"\nGenerating {full_lang} dataset...")

        try:
            generator = ImageGenerator(
                language=lang,
                sentence_count=sentences_per_image
            )         
            
            annotations = {
                "images": {},
                "categories": [
                    {
                        "id": 0,
                        "name": "text",
                        "supercategory": "text"
                    }
                ]
            }
            
            image_progress = tqdm(
                range(num_images),
                desc=f"{lang} 이미지 생성",
                position=1,
                leave=False
            )
            
            for i in image_progress:
                image_id = f"{lang}_image_{i+1}.jpg"
                try:
                    image_progress.set_postfix({
                        'image_id': image_id,
                        'sentences': sentences_per_image
                    })
                    
                    image, words = generator.generate_receipt()
                    
                    # 이미지 저장
                    image_path = f'{root_dir}/img/{image_id}'
                    image.save(image_path)
                    
                    # 단어 수 확인
                    word_count = len(words)
                    image_progress.set_postfix({
                        'image_id': image_id,
                        'words': word_count
                    })
                    
                    # UFO 형식으로 이미지 정보 저장
                    annotations["images"][image_id] = {
                        "words": {
                            str(idx): {
                                "transcription": word['text'],
                                "points": [
                                    [word['bbox'][i], word['bbox'][i+1]]
                                    for i in range(0, 8, 2)
                                ],
                                "language": lang,
                                "illegibility": False,
                                "orientation": "Horizontal",
                                "word_tags": None
                            }
                            for idx, word in enumerate(words)
                        },
                        "tags": None,
                        "license_tag": None
                    }
                    
                except Exception as e:
                    print(f"\nError generating image {image_id}: {str(e)}")
                    continue
            
            # JSON 파일 저장 경로 수정
            json_path = f"{root_dir}/ufo/train.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=4)
            
            print(f"\n{lang} 데이터셋 생성 완료:")
            print(f"- 생성된 이미지: {num_images}")
            print(f"- 저장 위치: data/{full_lang}/")
            
        except Exception as e:
            print(f"\nError generating dataset for {lang}: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Generate images and annotations from Wikipedia.")
    parser.add_argument('-n', '--num_images', type=int, default=10, help='Number of images to generate.')
    parser.add_argument('-s', '--sentences', type=int, default=10, help='Number of sentences to generate for each image.')

    args = parser.parse_args()

    languages = ['vi', 'th', 'ja', 'zh']
    generate_dataset(languages, 
                    num_images=args.num_images,
                    sentences_per_image=args.sentences
    )

if __name__ == "__main__":
    main()