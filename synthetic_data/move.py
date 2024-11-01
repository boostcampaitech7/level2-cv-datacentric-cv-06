import os
import shutil
import re

# 소스 및 대상 디렉토리 경로
src_dir = '/data/ephemeral/home/Dong_Yeong/raw_storie'
dst_dir = '/data/ephemeral/home/Dong_Yeong/level2-cv-datacentric-cv-06/code/data/storie/img'

# 대상 디렉토리가 없으면 생성
os.makedirs(dst_dir, exist_ok=True)

# 파일 이름에서 중복 표시 (1), (2) 등을 제거하는 함수
def get_original_filename(filename):
    # 확장자 제외한 파일 이름 추출
    name, ext = os.path.splitext(filename)
    # (숫자) 패턴 제거
    original_name = re.sub(r'\(\d+\)$', '', name)
    return original_name + ext

# 처리된 원본 파일명을 저장할 세트
processed_files = set()

# 소스 디렉토리의 모든 jpg 파일 처리
for filename in os.listdir(src_dir):
    if filename.lower().endswith('.jpg'):
        # 원본 파일명 얻기 (중복 표시 제거)
        original_filename = get_original_filename(filename)
        
        # 이미 처리된 원본 파일이 아닌 경우에만 복사
        if original_filename not in processed_files:
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, original_filename)
            
            # 파일 복사
            shutil.copy2(src_path, dst_path)
            processed_files.add(original_filename)

print(f'총 {len(processed_files)}개의 고유한 이미지가 복사되었습니다.')