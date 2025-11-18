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

    Args:
        json_data: 로드된 JSON 데이터

    Returns:
        어노테이션 리스트 (각 객체의 bbox, class 등)

    Note:
        실제 JSON 구조에 맞게 수정 필요
        예시 구조를 기반으로 작성됨
    """
    # TODO: 실제 데이터 다운로드 후 JSON 구조에 맞게 수정
    # 현재는 일반적인 COCO 스타일 가정

    annotations = []

    # 예시 1: COCO 형식
    if 'annotations' in json_data:
        for ann in json_data['annotations']:
            annotations.append({
                'category': ann.get('category', 'unknown'),
                'category_id': ann.get('category_id', -1),
                'bbox': ann.get('bbox', [0, 0, 0, 0]),  # [x, y, width, height]
                'attributes': ann.get('attributes', {})
            })

    # 예시 2: Custom 형식 (AI Hub 스타일)
    elif 'objects' in json_data:
        for obj in json_data['objects']:
            annotations.append({
                'category': obj.get('class', 'unknown'),
                'bbox': obj.get('bbox', [0, 0, 0, 0]),
                'attributes': obj.get('attributes', {})
            })

    return annotations


def get_image_info(json_data: Dict) -> Dict:
    """
    JSON 데이터에서 이미지 정보를 추출합니다.

    Args:
        json_data: 로드된 JSON 데이터

    Returns:
        이미지 정보 (파일명, 크기 등)
    """
    # TODO: 실제 JSON 구조에 맞게 수정

    if 'image' in json_data:
        return json_data['image']
    elif 'images' in json_data:
        return json_data['images']
    else:
        return {
            'file_name': json_data.get('file_name', 'unknown.jpg'),
            'width': json_data.get('width', 0),
            'height': json_data.get('height', 0)
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