"""
JSON 라벨링 데이터 로더
AI Hub 물류창고 안전 데이터셋의 JSON 파일을 파싱합니다.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import cv2


def load_json_labels(json_path: str) -> Dict:
    """
    JSON 라벨 파일을 로드합니다.

    Args:
        json_path: JSON 파일 경로

    Returns:
        파싱된 JSON 데이터 (dict)

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        json.JSONDecodeError: JSON 형식이 잘못되었을 때
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def load_image(image_path: str, mode: str = 'PIL') -> Optional[Image.Image]:
    """
    이미지 파일을 로드합니다.

    Args:
        image_path: 이미지 파일 경로
        mode: 'PIL' 또는 'cv2' (기본값: 'PIL')

    Returns:
        로드된 이미지 (PIL Image 또는 numpy array)

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    if mode == 'PIL':
        return Image.open(image_path).convert('RGB')
    elif mode == 'cv2':
        img = cv2.imread(image_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"지원하지 않는 모드입니다: {mode}. 'PIL' 또는 'cv2'를 사용하세요.")


def parse_annotations(json_data: Dict) -> List[Dict]:
    """
    JSON 데이터에서 어노테이션 정보를 추출합니다.
    AI Hub 물류창고 안전 데이터셋 형식에 맞게 파싱합니다.

    Args:
        json_data: 로드된 JSON 데이터

    Returns:
        어노테이션 리스트 (각 객체의 bbox, class 등)

    Note:
        JSON 구조:
        {
            "Learning data info.": {
                "annotation": [
                    {
                        "class_id": "SO-21",
                        "type": "box",
                        "coord": [x, y, width, height]
                    }
                ]
            }
        }
    """
    annotations = []

    # AI Hub 형식: Learning data info. -> annotation
    learning_info = json_data.get('Learning data info.', {})
    annotation_list = learning_info.get('annotation', [])

    for ann in annotation_list:
        # class_id와 coord 추출
        class_id = ann.get('class_id', 'unknown')
        bbox_type = ann.get('type', 'box')
        coord = ann.get('coord', [0, 0, 0, 0])

        # bbox가 유효한 경우만 추가
        if len(coord) == 4:
            annotations.append({
                'category': class_id,
                'category_id': class_id,
                'bbox': coord,  # [x, y, width, height]
                'type': bbox_type,
                'attributes': {}
            })

    return annotations


def get_image_info(json_data: Dict) -> Dict:
    """
    JSON 데이터에서 이미지 정보를 추출합니다.
    AI Hub 물류창고 안전 데이터셋 형식에 맞게 파싱합니다.

    Args:
        json_data: 로드된 JSON 데이터

    Returns:
        이미지 정보 (파일명, 크기 등)

    Note:
        JSON 구조:
        {
            "Source data Info.": {
                "source_data_ID": "L-211227_G19_I_UC-11_008_0144",
                "file_extension": "jpg"
            },
            "Raw data Info.": {
                "resolution": [1920, 1080]
            }
        }
    """
    # Source data Info에서 파일명 정보 추출
    source_info = json_data.get('Source data Info.', {})
    source_id = source_info.get('source_data_ID', 'unknown')
    file_ext = source_info.get('file_extension', 'jpg')

    # Raw data Info에서 해상도 정보 추출
    raw_info = json_data.get('Raw data Info.', {})
    resolution = raw_info.get('resolution', [0, 0])

    # 파일명 생성
    file_name = f"{source_id}.{file_ext}"

    return {
        'file_name': file_name,
        'width': resolution[0] if len(resolution) >= 1 else 0,
        'height': resolution[1] if len(resolution) >= 2 else 0,
        'source_data_ID': source_id,
        'file_extension': file_ext
    }


def scan_dataset(data_dir: str, ext: str = '.json') -> List[str]:
    """
    데이터셋 디렉토리를 스캔하여 모든 라벨 파일을 찾습니다.

    Args:
        data_dir: 데이터셋 루트 디렉토리
        ext: 찾을 파일 확장자 (기본값: '.json')

    Returns:
        파일 경로 리스트
    """
    data_path = Path(data_dir)
    files = list(data_path.rglob(f'*{ext}'))
    return [str(f) for f in files]


if __name__ == "__main__":
    # 테스트 코드
    print("데이터 로더 테스트")
    print("실제 데이터 다운로드 후 테스트하세요.")