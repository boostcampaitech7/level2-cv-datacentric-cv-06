import numpy as np
import albumentations as A

class RectangleShadowTransform(A.ImageOnlyTransform):
    def __init__(self, 
                 opacity_range=(0.5, 0.7), 
                 width_range=(0.25, 0.5), 
                 height_range=(0.25, 0.5), 
                 p=0.5):
        """
        이미지에 사각형 모양의 반투명한 그림자를 추가하는 함수
        
        Parameters:
        -----------
        image : numpy.ndarray
            입력 이미지 (BGR 또는 RGB 형식)
        opacity_range : tuple
            그림자의 투명도 범위 (min, max), 0에 가까울수록 어두움, 1에 가까울수록 밝음
        width_range : tuple
            그림자 너비의 범위 (이미지 너비 대비 비율)
        height_range : tuple
            그림자 높이의 범위 (이미지 높이 대비 비율)
        p=0.5 : float
            적용 확률
            
        Returns:
        --------
        numpy.ndarray
            그림자가 추가된 이미지
        """
        super().__init__(always_apply=False, p=p)
        self.opacity_range = opacity_range
        self.width_range = width_range
        self.height_range = height_range

    def apply(self, image, **params):
        height, width = image.shape[:2]
        
        # 그림자 크기 계산
        shadow_width = int(width * np.random.uniform(self.width_range[0], self.width_range[1]))
        shadow_height = int(height * np.random.uniform(self.height_range[0], self.height_range[1]))
        
        # 그림자 위치 계산
        x1 = np.random.randint(0, width - shadow_width)
        y1 = np.random.randint(0, height - shadow_height)
        x2 = x1 + shadow_width
        y2 = y1 + shadow_height
        
        # 그림자 투명도 설정
        opacity = np.random.uniform(self.opacity_range[0], self.opacity_range[1])
        
        # 결과 이미지 생성
        result = image.copy()
        result[y1:y2, x1:x2] = (result[y1:y2, x1:x2] * opacity).astype(np.uint8)
        
        return result