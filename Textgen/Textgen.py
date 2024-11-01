import os
import argparse
import json
import matplotlib.pyplot as plt
from trdg.generators import GeneratorFromWikipedia, GeneratorFromStrings
import treescope
import numpy as np
from synth_utils import *
import cv2

class ImageGenerator:
    def __init__(self, count=128, language='ko', fonts=['fonts/NotoSans-Italic-VariableFont_wdth,wght.ttf']):
        self.count = count
        self.language = language
        self.fonts = fonts
        self.annotations = {}  # Dictionary to hold annotation data for JSON

    def get_words(self):
        # 위키피디아로부터 텍스트 생성
        generator = GeneratorFromWikipedia(
            language=self.language,
            count=self.count,
            fonts=self.fonts,
            size=64,
        )

        #하나의 문장으로 합침
        strings = []
        for _, (patch, text) in enumerate(generator):
            strings.append(text)

        # 공백 기준으로 단어로 쪼갬
        total_str = " ".join(strings)
        paragrphs = " ".join([x.strip() for x in total_str])
        sentences = [x.strip() for x in total_str.split(" ")]
        sentences = [x for x in sentences if x]
        sentences_len = len(sentences)

        #단어 기반으로 Text생성
        word_generator = GeneratorFromStrings(
            sentences,
            language=self.language,
            size=64,
            count=sentences_len,
            fonts=self.fonts,
            margins=(5, 5, 5, 5),
        )

        #하나의 단어 list로 합침
        words = []
        for _, (patch, text) in enumerate(word_generator):
            words.append({"patch": patch, "text": text, "size": patch.size, "margin": 5, "bbox": patch.getbbox()})
        
        return words
    #이미지 변형
    def your_vm_function(self, bbox: list[float], M: np.ndarray):
        v = np.array(bbox).reshape(-1, 2).T
        v = np.vstack([v, np.ones((1, 4))])
        v = np.dot(M,v)
        v = np.array([v[0]/v[2],v[1]/v[2]])
        out = v.T.flatten().tolist()
        
        return out
    
    def perturb_document_inplace(self, document: Document, pad=0, color=None):
        if color is None:
            color = [64, 64, 64]
        width, height = np.array(document["image"].size)
        magnitude_lb = 0
        magnitude_ub = 200
        src = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
        perturb = np.random.uniform(magnitude_lb, magnitude_ub, (4, 2)) * np.array(
            [[1, 1], [-1, 1], [-1, -1], [1, -1]]
        )
        perturb = perturb.astype(np.float32)
        dst = src + perturb

        # obtain the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)

        # transform the image
        out = cv2.warpPerspective(
            np.array(document["image"]),
            M,
            document["image"].size,
            flags=cv2.INTER_LINEAR,
            borderValue=color,
        )
        out = Image.fromarray(out)
        document["image"] = out

        # transform the bounding boxes
        for word in document["words"]:
            bbox = word["bbox"]

            word["bbox"] = self.your_vm_function(bbox, M)
        return document

    #이미지, Annotation저장
    def save_images_and_annotations(self, num_images=10):
        data_root = 'data/'+self.language
        os.makedirs(data_root +'/images', exist_ok=True)
        os.makedirs(data_root +'/json', exist_ok=True)

        for i in range(num_images):
            # Generate document using make_document function
            document = self.perturb_document_inplace(make_document(self.get_words()))
            image_id = f"{self.language}_image_{i+1}"
            print(f"{image_id}.jpg Generating")
            self.annotations[image_id] = {
                "paragraphs": {},
                "words": {},
                "chars": {},
                "img_w": document["image"].size[0],
                "img_h": document["image"].size[1],
                "num_patches": None,
                "tags": [],
                "relations": {},
                "annotation_log": {},
                "license_tag": {}
            }

            for idx, word in enumerate(document["words"]):
                bbox = word['bbox']
                self.annotations[image_id]["words"][f"{idx+1:04}"] = {
                    "transcription": word['text'],
                    "points": [
                        [bbox[0], bbox[1]],  # x1, y1 (좌상단)
                        [bbox[2], bbox[3]],  # x2, y2 (우상단)
                        [bbox[4], bbox[5]],  # x3, y3 (우하단)
                        [bbox[6], bbox[7]],  # x4, y4 (좌하단)
                    ]
                }

            # Save image file
            image_path = os.path.join(data_root, f"images/{image_id}.jpg")
            document["image"].save(image_path)

        # Save annotations to JSON
        annotation_path = os.path.join(data_root, "json/annotations.json")
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=4)

    def run(self, num_images=10):
        self.save_images_and_annotations(num_images)
    

def main():
    parser = argparse.ArgumentParser(description="Generate images and annotations from Wikipedia.")
    parser.add_argument('-c', '--count', type=int, default=128, help='Number of sentences to generate for each image.')
    parser.add_argument('-l', '--language', type=str, default='ko', help='Language code for the Wikipedia sentences.')
    parser.add_argument('-n', '--num_images', type=int, default=10, help='Number of images to generate.')
    parser.add_argument('-f', '--fonts', type=str, nargs='+', default=['fonts/NotoSans-Italic-VariableFont_wdth,wght.ttf'], help='List of font paths.')

    args = parser.parse_args()

    generator = ImageGenerator(count=args.count, language=args.language, fonts=args.fonts)
    generator.run(num_images=args.num_images)

    print(f"Generated Total {args.num_images} {args.language} Images is Done!!")
if __name__ == "__main__":
    main()
