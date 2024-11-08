 # Boostcamp AI Tech 7 CV 06
 
## 다국어 영수증 OCR
### 2024.10.30 10:00 ~ 2024.11.07 19:00


![image](https://github.com/user-attachments/assets/3d11bf4f-c77b-4e18-b7d3-c5c97a740cee)
## Description
카메라로 영수증을 인식할 경우 자동으로 영수증 내용이 입력되는 어플리케이션이 있습니다.

이처럼 OCR (Optical Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

OCR은 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.

본 대회에서는 다국어 (중국어, 일본어, 태국어, 베트남어)로 작성된 영수증 이미지에 대한 OCR task를 수행합니다.

본 대회에서는 글자 검출만을 수행합니다. 즉, 이미지에서 어떤 위치에 글자가 있는지를 예측하는 모델을 제작합니다.

본 대회는 제출된 예측 (prediction) 파일로 평가합니다.

대회 기간과 task 난이도를 고려하여 코드 작성에 제약사항이 있습니다. 상세 내용은 Data > Baseline Code (베이스라인 코드)에 기술되어 있습니다.

모델의 입출력 형식은 다음과 같습니다.

입력 : 글자가 포함된 JPG 이미지 (학습 총 400장, 테스트 총 120장)

출력 : bbox 좌표가 포함된 UFO Format (상세 제출 형식은 Overview > Metric 탭 및 강의 6강 참조)

## Result
![image](https://github.com/user-attachments/assets/7a6d4b29-4691-496a-9b72-7a05f849d672)
최종 Private F1-score 0.8799 달성

## Contributor
| [![](https://avatars.githubusercontent.com/jhuni17)](https://github.com/jhuni17) | [![](https://avatars.githubusercontent.com/jung0228)](https://github.com/jung0228) | [![](https://avatars.githubusercontent.com/Jin-SukKim)](https://github.com/Jin-SukKim) | [![](https://avatars.githubusercontent.com/kimdyoc13)](https://github.com/kimdyoc13) | [![](https://avatars.githubusercontent.com/MinSeok1204)](https://github.com/MinSeok1204) | [![](https://avatars.githubusercontent.com/airacle100)](https://github.com/airacle100) |
| ---------------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| [최재훈](https://github.com/jhuni17)                  | [정현우](https://github.com/jung0228)                  | [김진석](https://github.com/Jin-SukKim)                  | [김동영](https://github.com/kimdyoc13)                  | [최민석](https://github.com/MinSeok1204)                  | [윤정우](https://github.com/airacle100)                  |


## Requirements
```
lanms==1.0.2
opencv-python==4.10.0.84
shapely==2.0.5
albumentations==1.4.12
torch==2.1.0
tqdm==4.66.5
albucore==0.0.13
annotated-types==0.7.0
contourpy==1.1.1
cycler==0.12.1
eval_type_backport==0.2.0
filelock==3.15.4
fonttools==4.53.1
fsspec==2024.6.1
imageio==2.35.0
importlib_resources==6.4.2
Jinja2==3.1.4
kiwisolver==1.4.5
lazy_loader==0.4
MarkupSafe==2.1.5
matplotlib==3.7.5
mpmath==1.3.0
networkx==3.1
numpy==1.24.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.18.1
nvidia-nvjitlink-cu12==12.6.20
nvidia-nvtx-cu12==12.1.105
packaging==24.1
pillow==10.4.0
pydantic==2.8.2
pydantic_core==2.20.1
pyparsing==3.1.2
python-dateutil==2.9.0.post0
PyWavelets==1.4.1
PyYAML==6.0.2
scikit-image==0.21.0
scipy==1.10.1
six==1.16.0
sympy==1.13.2
tifffile==2023.7.10
tomli==2.0.1
triton==2.1.0
typing_extensions==4.12.2
```
