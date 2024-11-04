import os
import glob

# 지정된 디렉토리 경로
directory = '/data/ephemeral/home/Dong_Yeong/level2-cv-datacentric-cv-06/code/data/storie(626)'

# '(' 문자가 포함된 모든 파일/디렉토리 찾기
files_with_parentheses = glob.glob(os.path.join(directory, '*(*)*'))

# 찾은 파일들을 순회하며 삭제
for file_path in files_with_parentheses:
    try:
        if os.path.isfile(file_path):  # 파일인 경우
            os.remove(file_path)
        elif os.path.isdir(file_path):  # 디렉토리인 경우
            os.rmdir(file_path)  # 디렉토리가 비어있을 경우만 삭제됨
        print(f"삭제됨: {file_path}")
    except Exception as e:
        print(f"삭제 실패: {file_path}, 에러: {str(e)}")

print("삭제 작업 완료")