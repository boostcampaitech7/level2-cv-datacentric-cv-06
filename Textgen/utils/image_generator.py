import cv2
import numpy as np
from PIL import Image
from .text_generator import TextGenerator
from .receipt_layout import ReceiptLayoutGenerator
import random
from .synth_utils import *


class ImageGenerator:
    def __init__(self, language: str, width: int = 800, height: int = 1200, sentence_count: int = 128):
        self.text_generator = TextGenerator(language, count=sentence_count)
        self.layout_generator = ReceiptLayoutGenerator(width, height)
        self.width = width
        self.height = height

    def generate_receipt(self):
        try:
            # 텍스트 생성
            texts = self.text_generator.generate_from_wikipedia()
            if not texts:
                raise ValueError("No texts generated")
                
            # 단어 이미지 생성
            word_images = self.text_generator.generate_word_images(texts)
            if not word_images:
                raise ValueError("No word images generated")
                
            # 레이아웃 생성
            image, placed_words = self.layout_generator.create_layout(word_images)
            
            # 이미지 후처리
            #processed_image = self._apply_receipt_effects(image)
            processed_image, placed_words = self.perturb_document_inplace(image, placed_words)      
            return processed_image, placed_words
            
        except Exception as e:
            print(f"Error in generate_receipt: {str(e)}")
            raise

    def _apply_receipt_effects(self, image):
        """영수증 효과 적용"""
        try:
            # PIL Image를 numpy array로 변환
            img_array = np.array(image)
            
            # 노이즈 추가
            noise = np.random.normal(0, 2, img_array.shape).astype(np.uint8)
            img_array = cv2.add(img_array, noise)
            
            # 약간의 블러
            img_array = cv2.GaussianBlur(img_array, (3, 3), 0.5)
            
            # 미세한 회전
            angle = np.random.uniform(-0.5, 0.5)
            center = (self.width // 2, self.height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_array = cv2.warpAffine(
                img_array, 
                rotation_matrix, 
                (self.width, self.height),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )
            
            # numpy array를 PIL Image로 변환
            return Image.fromarray(img_array)
            
        except Exception as e:
            print(f"Error in _apply_receipt_effects: {str(e)}")
            return image  # 에러 발생 시 원본 이미지 반환       

    def your_vm_function(self, bbox: list[float], M: np.ndarray):
        v = np.array(bbox).reshape(-1, 2).T
        v = np.vstack([v, np.ones((1, 4))])
        v = np.dot(M,v)
        v = np.array([v[0]/v[2],v[1]/v[2]])
        out = v.T.flatten().tolist()
        
        return out
    
    def perturb_document_inplace(self, processed_image, placed_words, pad=0, color=None):
        if color is None:
            color = [64, 64, 64]
        width, height = np.array(processed_image.size)
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
            np.array(processed_image),
            M,
            processed_image.size,
            flags=cv2.INTER_LINEAR,
            borderValue=color,
        )
        out = Image.fromarray(out)
        processed_image = out

        # transform the bounding boxes
        for word in placed_words:
            bbox = word["bbox"]

            word["bbox"] = self.your_vm_function(bbox, M)
        return processed_image, placed_words