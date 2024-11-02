import streamlit as st
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

def read_prediction_csv(csv_path):
    """CSV 파일 내의 JSON 데이터를 읽는 함수"""
    try:
        # 파일을 텍스트로 읽기
        with open(csv_path, 'r', encoding='utf-8') as f:
            json_str = f.read()
        
        # JSON 파싱
        pred_data = json.loads(json_str)
        
        # 예측 결과를 DataFrame 형식으로 변환
        predictions = []
        for img_id, img_data in pred_data['images'].items():
            for word_id, word_info in img_data['words'].items():
                # points를 쉼표로 구분된 문자열로 변환
                points_str = ','.join([str(coord) for point in word_info['points'] for coord in point])
                predictions.append({
                    'image_id': img_id,
                    'points': points_str,
                    'text': word_info.get('text', '')
                })
        
        return pd.DataFrame(predictions)
    except Exception as e:
        st.error(f"Error reading prediction file: {str(e)}")
        return None

def get_dataset_name(path):
    path = Path(path)
    if 'filtered' in str(path):
        return f"{path.parent.parent.parent.name} - {path.name}"
    return f"{path.parent.parent.name} - {path.name}"

def get_image_paths(base_path, json_paths, data_type):
    image_paths = {}
    for json_path in json_paths:
        # filtered 데이터의 경우 경로 조정
        json_path = Path(json_path)
        if 'filtered' in str(json_path):
            img_dir = json_path.parent.parent.parent / 'img' / data_type.lower()
        else:
            img_dir = json_path.parent.parent / 'img' / data_type.lower()
        
        json_data = read_json(str(json_path))
        for img_id in json_data['images'].keys():
            img_path = str(img_dir / img_id)
            if Path(img_path).exists():
                image_paths[img_id] = img_path
            else:
                st.warning(f"Image not found: {img_path}")
    return image_paths

def visualize_image(img_id, img_info, img_path, is_train=True):
    img = cv2.imread(img_path)
    if img is None:
        st.error(f"Cannot load image: {img_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if is_train:
        # 바운딩 박스 그리기
        for word_id, word in img_info['words'].items():
            if not word.get('illegibility', False):
                transcription = word.get('transcription')
                points = np.array(word.get('points', []))
                
                if len(points) > 0:
                    # 빈 텍스트인 경우 빨간색, 아닌 경우 파란색으로 표시
                    is_empty = transcription is not None and transcription.strip() == ''
                    color = (255, 0, 0) if is_empty else (0, 0, 255)  # BGR
                    
                    # 박스 그리기
                    cv2.polylines(img, [points.astype(np.int32)], True, color, 2)
    
    return img

def visualize_comparison(img_id, original_info, filtered_info, img_path):
    """원본과 필터링된 이미지를 나란히 비교하는 함수"""
    img = cv2.imread(img_path)
    if img is None:
        st.error(f"Cannot load image: {img_path}")
        return
    
    img_original = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    img_filtered = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    # 원본 이미지에 박스 그리기 (수정된 부분)
    for word_id, word in original_info['words'].items():
        if not word.get('illegibility', False):
            transcription = word.get('transcription')
            points = np.array(word.get('points', []))
            if len(points) > 0:
                # 빈 텍스트인 경우 빨간색, 아닌 경우 파란색으로 표시
                is_empty = transcription is not None and transcription.strip() == ''
                color = (255, 0, 0) if is_empty else (0, 0, 255)
                cv2.polylines(img_original, [points.astype(np.int32)], True, color, 2)
    
    # 필터링된 이미지에 박스 그리기
    for word_id, word in filtered_info['words'].items():
        if not word.get('illegibility', False):
            transcription = word.get('transcription')
            points = np.array(word.get('points', []))
            if len(points) > 0:
                is_empty = transcription is not None and transcription.strip() == ''
                color = (255, 0, 0) if is_empty else (0, 0, 255)
                cv2.polylines(img_filtered, [points.astype(np.int32)], True, color, 2)
    
    return img_original, img_filtered
def visualize_prediction(img, img_id, pred_df):
    """예측 결과로 박스를 그리는 함수"""
    img_pred = img.copy()
    
    # 현재 이미지에 대한 예측 결과 필터링
    img_predictions = pred_df[pred_df['image_id'] == img_id]
    
    if len(img_predictions) == 0:
        st.warning(f"No predictions found for image {img_id}")
        return img_pred
    
    for _, row in img_predictions.iterrows():
        try:
            # points 문자열을 좌표 배열로 변환
            points_str = row['points']
            coords = [float(x) for x in points_str.split(',')]
            
            # 4개의 점으로 재구성 (x,y 쌍)
            points = np.array([[coords[i], coords[i+1]] for i in range(0, len(coords), 2)])
            
            # 예측된 텍스트가 비어있는지 확인
            is_empty = pd.isna(row['text']) or row['text'].strip() == ''
            color = (255, 0, 0) if is_empty else (0, 0, 255)  # BGR
            
            # 박스 그리기
            cv2.polylines(img_pred, [points.astype(np.int32)], True, color, 2)
        except Exception as e:
            st.warning(f"Error processing points for prediction in image {img_id}: {str(e)}")
    
    return img_pred

def main():
    st.title("Receipt Image Visualization")
    
    base_path = "../data"
    
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    if 'prev_data_type' not in st.session_state:
        st.session_state.prev_data_type = None
    if 'prev_dataset' not in st.session_state:
        st.session_state.prev_dataset = None
    
    data_type = st.radio("Select Data Type:", ["Train", "Test"])
    
    # 수정된 부분: Filtered 데이터 선택 로직
    use_filtered = False
    if data_type == "Train":
        use_filtered = st.checkbox("Use Filtered Data")

    # JSON 파일 패턴 설정
    if use_filtered:
        # filtered 폴더 내의 모든 JSON 파일을 찾음
        json_pattern = f"*_receipt/ufo/filtered/*.json"
    else:
        json_pattern = f"*_receipt/ufo/*{data_type.lower()}.json"
    
    json_paths = glob.glob(str(Path(base_path) / json_pattern))
    
    # Test 선택 시 CSV 파일 선택 옵션 추가
    pred_df = None
    if data_type == "Test":
        csv_files = glob.glob("./csv/*.csv")
        if csv_files:
            selected_csv = st.selectbox(
                "Select Prediction CSV:",
                csv_files,
                format_func=lambda x: Path(x).name
            )
            pred_df = read_prediction_csv(selected_csv)
    
    # 데이터 타입이 변경되면 인덱스 초기화
    if st.session_state.prev_data_type != data_type:
        st.session_state.current_idx = 0
        st.session_state.prev_data_type = data_type
    
    # JSON 파일 패턴 설정 (중복 제거)
    if use_filtered:
        # filtered 폴더 내의 모든 JSON 파일을 찾음
        json_pattern = f"*_receipt/ufo/filtered/*.json"
    else:
        json_pattern = f"*_receipt/ufo/*{data_type.lower()}.json"
    
    json_paths = glob.glob(str(Path(base_path) / json_pattern))
    
    if not json_paths:
        st.error(f"No {'filtered ' if use_filtered else ''}{data_type} JSON files found!")
        return
    
    # 데이터셋 선택 (session state로 관리)
    if 'prev_dataset' not in st.session_state:
        st.session_state.prev_dataset = None
        
    selected_json = st.selectbox(
        "Select Dataset:",
        json_paths,
        format_func=get_dataset_name
    )
    
    # 데이터셋이 변경되면 인덱스 초기화
    if st.session_state.prev_dataset != selected_json:
        st.session_state.current_idx = 0
        st.session_state.prev_dataset = selected_json
    
    # JSON 데이터 로드
    data = read_json(selected_json)
    
    # 이미지 경로 매핑
    image_paths = get_image_paths(base_path, [selected_json], data_type)
    
    # 이미지 목록
    img_ids = list(data['images'].keys())
    
    if not img_ids:
        st.error("No images found in the selected dataset!")
        return
    
    # 현재 이미지 인덱스를 세션 스테이트로 관리
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    
    # 인덱스가 범위를 벗어나면 조정
    st.session_state.current_idx = min(st.session_state.current_idx, len(img_ids) - 1)
    
    # 이전/다음 버튼 컬럼
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        if st.button('⬅️ Prev') and len(img_ids) > 0:
            st.session_state.current_idx = (st.session_state.current_idx - 1) % len(img_ids)
            
    with col3:
        if st.button('Next ➡️') and len(img_ids) > 0:
            st.session_state.current_idx = (st.session_state.current_idx + 1) % len(img_ids)
    
    # 현재 이미지 정보
    current_img_id = img_ids[st.session_state.current_idx]
    
    # 현재 이미지 번호 표시
    with col2:
        st.write(f"Image {st.session_state.current_idx + 1} of {len(img_ids)}")
    
    img_info = data['images'][current_img_id]
    img_path = image_paths.get(current_img_id)
    
    if img_path:
        # 이미지 정보 표시
        st.write("Image Information:")
        st.write(f"- Size: {img_info['img_w']} x {img_info['img_h']}")
        st.write(f"- Tags: {img_info.get('tags', ['None'])}")
        st.write(f"- Total Words: {len(img_info['words'])}")
        
        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            st.error(f"Cannot load image: {img_path}")
            return
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if data_type == "Test" and pred_df is not None:
            # Test 이미지일 경우 원본과 예측 결과를 나란히 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Original Test Image")
                st.image(img, use_column_width=True)
                
            with col2:
                st.write("Prediction Result")
                img_pred = visualize_prediction(img, current_img_id, pred_df)
                st.image(img_pred, use_column_width=True)
        elif data_type == "Train" and use_filtered:
            # 원본 train.json 파일 경로
            original_json_path = str(Path(selected_json).parent.parent / f"{data_type.lower()}.json")
            original_data = read_json(original_json_path)
            
            # 이미지 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Original Train Image")
                img_original, img_filtered = visualize_comparison(
                    current_img_id,
                    original_data['images'][current_img_id],
                    data['images'][current_img_id],
                    img_path
                )
                st.image(img_original, use_column_width=True)
                
            with col2:
                st.write("Filtered Train Image")
                st.image(img_filtered, use_column_width=True)
        else:
            # Train 이미지는 박스 표시와 함께 표시
            img_with_boxes = visualize_image(current_img_id, img_info, img_path, 
                                           is_train=(data_type.lower() == 'train'))
            st.image(img_with_boxes, caption=f"{data_type} Image: {current_img_id}", 
                    use_column_width=True)
    else:
        st.error("Image file not found!")

if __name__ == "__main__":
    main()