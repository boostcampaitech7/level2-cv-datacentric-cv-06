import cv2
import numpy as np
from typing import Tuple, Dict, Optional, Union, List
import random
from scipy.ndimage import zoom
import albumentations as A

class ReceiptAugmentor:
    """영수증 OCR을 위한 bbox 좌표 변환을 지원하는 data augmentation 클래스"""
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (800, 1000),
                 random_state: Optional[int] = None):
        """
        Args:
            image_size: 기본 이미지 크기 (width, height)
            random_state: 랜덤 시드
        """
        self.image_size = image_size
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            
        # Albumentations 변환 파이프라인 설정
        self.base_transform = A.Compose([
            A.OneOf([
                A.RandomBrightness(limit=0.2, p=0.5),
                A.RandomContrast(limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.5),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
            ], p=0.3),
        ])

    def _transform_bbox_points(self, 
                             points: np.ndarray, 
                             transform_matrix: np.ndarray) -> np.ndarray:
        """bbox 좌표를 변환 행렬을 사용하여 변환

        Args:
            points: [N, 8] 형태의 bbox 좌표 배열 (x1,y1,x2,y2,x3,y3,x4,y4)
            transform_matrix: 3x3 변환 행렬

        Returns:
            변환된 bbox 좌표 배열
        """
        points = points.reshape(-1, 4, 2)  # [N, 4, 2] 형태로 변환
        transformed_points = []

        for bbox in points:
            # 동차 좌표로 변환
            homogeneous_points = np.hstack([bbox, np.ones((4, 1))])
            
            # 변환 적용
            transformed = transform_matrix @ homogeneous_points.T
            transformed = transformed / transformed[2]
            transformed = transformed[:2].T
            
            # 원래 형태로 변환
            transformed_points.append(transformed.reshape(-1))

        return np.array(transformed_points)

    def _apply_perspective_transform(self, 
                                   image: np.ndarray,
                                   bboxes: np.ndarray,
                                   max_shift: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """투시 변환 적용"""
        height, width = image.shape[:2]
        
        # 기준 포인트
        src_pts = np.float32([
            [0, 0],
            [width-1, 0],
            [width-1, height-1],
            [0, height-1]
        ])
        
        # 랜덤한 변형 추가
        max_shift_px = min(height, width) * max_shift
        dst_pts = src_pts + np.random.uniform(
            -max_shift_px,
            max_shift_px,
            src_pts.shape
        )
        
        # 변환 행렬 계산 및 적용
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        transformed_image = cv2.warpPerspective(image, matrix, (width, height))
        
        # bbox 좌표 변환
        transformed_bboxes = self._transform_bbox_points(bboxes, matrix)
        
        return transformed_image, transformed_bboxes

    def _apply_rotation(self, 
                       image: np.ndarray,
                       bboxes: np.ndarray,
                       angle_range: Tuple[float, float] = (-5, 5)) -> Tuple[np.ndarray, np.ndarray]:
        """회전 변환 적용"""
        height, width = image.shape[:2]
        angle = random.uniform(*angle_range)
        
        # 회전 중심점
        center = (width / 2, height / 2)
        
        # 회전 행렬 계산
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotation_matrix = np.vstack([rotation_matrix, [0, 0, 1]])  # 3x3 행렬로 변환
        
        # 이미지 회전
        rotated_image = cv2.warpAffine(
            image,
            rotation_matrix[:2],
            (width, height)
        )
        
        # bbox 좌표 회전
        rotated_bboxes = self._transform_bbox_points(bboxes, rotation_matrix)
        
        return rotated_image, rotated_bboxes

    def _apply_scale(self, 
                    image: np.ndarray,
                    bboxes: np.ndarray,
                    scale_range: Tuple[float, float] = (0.9, 1.1)) -> Tuple[np.ndarray, np.ndarray]:
        """크기 변환 적용"""
        height, width = image.shape[:2]
        scale = random.uniform(*scale_range)
        
        # 스케일링 행렬 생성
        scale_matrix = np.array([
            [scale, 0, width * (1 - scale) / 2],
            [0, scale, height * (1 - scale) / 2],
            [0, 0, 1]
        ])
        
        # 이미지 스케일링
        scaled_image = cv2.warpAffine(
            image,
            scale_matrix[:2],
            (width, height)
        )
        
        # bbox 좌표 스케일링
        scaled_bboxes = self._transform_bbox_points(bboxes, scale_matrix)
        
        return scaled_image, scaled_bboxes

    def augment(self,
                image: np.ndarray,
                bboxes: np.ndarray,
                augmentation_types: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        지정된 증강 기법들을 적용하고 bbox 좌표도 함께 변환
        
        Args:
            image: 입력 이미지
            bboxes: [N, 8] 형태의 bbox 좌표 배열 (x1,y1,x2,y2,x3,y3,x4,y4)
            augmentation_types: 적용할 증강 기법 리스트
                ['perspective', 'rotate', 'scale']
        
        Returns:
            증강된 이미지와 변환된 bbox 좌표
        """
        if augmentation_types is None:
            augmentation_types = ['perspective', 'rotate', 'scale']
            
        result = image.copy()
        result_bboxes = bboxes.copy()
        
        # 기본 변환 적용 (밝기, 대비 등)
        result = self.base_transform(image=result)['image']
        
        # geometric 변환 적용
        for aug_type in augmentation_types:
            if aug_type == 'perspective':
                result, result_bboxes = self._apply_perspective_transform(
                    result, result_bboxes
                )
            elif aug_type == 'rotate':
                result, result_bboxes = self._apply_rotation(
                    result, result_bboxes
                )
            elif aug_type == 'scale':
                result, result_bboxes = self._apply_scale(
                    result, result_bboxes
                )
                
        return result, result_bboxes
    
    def generate_variations(self,
                          image: np.ndarray,
                          bboxes: np.ndarray,
                          num_variations: int = 5,
                          augmentation_types: List[str] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """여러 개의 증강된 이미지와 bbox 좌표 생성"""
        variations = []
        for _ in range(num_variations):
            aug_image, aug_bboxes = self.augment(
                image, bboxes, augmentation_types
            )
            variations.append((aug_image, aug_bboxes))
        return variations

    def visualize_bboxes(self,
                        image: np.ndarray,
                        bboxes: np.ndarray,
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
        """bbox를 시각화"""
        vis_image = image.copy()
        
        for bbox in bboxes:
            pts = bbox.reshape(4, 2).astype(np.int32)
            cv2.polylines(
                vis_image,
                [pts],
                isClosed=True,
                color=color,
                thickness=thickness
            )
            
        return vis_image

def demo_bbox_augmentation():
    """bbox 변환 포함한 증강 데모"""
    # 이미지 로드
    image = cv2.imread('receipt.jpg')
    
    # 샘플 bbox 좌표 (예시)
    bboxes = np.array([
        [100, 100, 200, 100, 200, 150, 100, 150],  # 첫 번째 단어
        [300, 200, 400, 200, 400, 250, 300, 250],  # 두 번째 단어
    ])
    
    # Augmentor 초기화
    augmentor = ReceiptAugmentor()
    
    # 증강 적용
    augmented_image, augmented_bboxes = augmentor.augment(
        image,
        bboxes,
        augmentation_types=['perspective', 'rotate', 'scale']
    )
    
    # 결과 시각화
    original_vis = augmentor.visualize_bboxes(image, bboxes)
    augmented_vis = augmentor.visualize_bboxes(augmented_image, augmented_bboxes)
    
    # 결과 저장
    cv2.imwrite('original_with_bbox.jpg', original_vis)
    cv2.imwrite('augmented_with_bbox.jpg', augmented_vis)

if __name__ == '__main__':
    demo_bbox_augmentation()
    
    
'''
주요 변경 및 추가된 기능:

Geometric 변환 지원:

Perspective Transform
Rotation
Scaling
각 변환에서 bbox 좌표도 함께 변환됩니다.


bbox 좌표 변환:

_transform_bbox_points: 모든 기하학적 변환에 대해 bbox 좌표를 변환
동차 좌표계를 사용하여 정확한 변환 지원


시각화 기능:

visualize_bboxes: 원본/증강된 이미지와 bbox를 시각화



사용 예시:
pythonCopy# Augmentor 초기화
augmentor = ReceiptAugmentor()

# bbox 좌표 준비 (예시)
bboxes = np.array([
    [x1, y1, x2, y2, x3, y3, x4, y4],  # 첫 번째 단어
    [x1, y1, x2, y2, x3, y3, x4, y4],  # 두 번째 단어
    ...
])

# 단일 증강
augmented_image, augmented_bboxes = augmentor.augment(
    image,
    bboxes,
    augmentation_types=['perspective', 'rotate', 'scale']
)

# 시각화
vis_image = augmentor.visualize_bboxes(augmented_image, augmented_bboxes)

# 여러 변형 생성
variations = augmentor.generate_variations(
    image,
    bboxes,
    num_variations=5
)
주의사항:

perspective transform 시 bbox가 이미지 경계를 벗어나지 않도록 max_shift 파라미터 조정
회전 변환 시 angle_range를 적절히 조정하여 텍스트가 너무 기울어지지 않도록 설정
스케일링 시 scale_range를 적절히 조정하여 텍스트가 읽을 수 없을 정도로 작아지거나 커지지 않도록 설정

개선 가능한 부분:

더 복잡한 기하학적 변환 추가
bbox 유효성 검사 기능 추가
멀티스레딩을 통한 성능 최적화
다양한 augmentation 조합에 대한 presets 추가

필요한 기능이나 수정하고 싶은 부분이 있다면 말씀해 주세요.
'''