"""
데이터 유효성 검증 모듈
이미지와 라벨 데이터의 무결성을 확인합니다.
"""

import os
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np


class DataValidator:
    """데이터셋 유효성 검사기"""

    def __init__(self, min_bbox_size: int = 10):
        """
        Args:
            min_bbox_size: 최소 bbox 크기 (픽셀)
        """
        self.min_bbox_size = min_bbox_size
        self.errors = []
        self.warnings = []

    def validate_image(self, image_path: str) -> bool:
        """
        이미지 파일 유효성 검사

        Args:
            image_path: 이미지 파일 경로

        Returns:
            유효하면 True, 아니면 False
        """
        # 파일 존재 확인
        if not os.path.exists(image_path):
            self.errors.append(f"이미지 파일 없음: {image_path}")
            return False

        # 이미지 로드 가능 확인
        try:
            img = Image.open(image_path)
            img.verify()  # 파일 손상 확인
            img = Image.open(image_path)  # verify 후 다시 열기
            img.load()  # 실제 로드
        except Exception as e:
            self.errors.append(f"이미지 로드 실패 {image_path}: {str(e)}")
            return False

        # 이미지 크기 확인
        width, height = img.size
        if width < 32 or height < 32:
            self.warnings.append(f"이미지가 너무 작음 {image_path}: {width}x{height}")

        return True

    def validate_bbox(
        self,
        bbox: List[float],
        image_width: int,
        image_height: int,
        image_path: str = ""
    ) -> bool:
        """
        Bounding box 유효성 검사

        Args:
            bbox: [x, y, width, height] 형식의 bbox
            image_width: 이미지 너비
            image_height: 이미지 높이
            image_path: 에러 메시지용 이미지 경로

        Returns:
            유효하면 True, 아니면 False
        """
        if len(bbox) != 4:
            self.errors.append(f"잘못된 bbox 형식 {image_path}: {bbox}")
            return False

        x, y, w, h = bbox

        # bbox가 이미지 범위 내에 있는지 확인
        if x < 0 or y < 0 or x + w > image_width or y + h > image_height:
            self.errors.append(
                f"bbox가 이미지 범위를 벗어남 {image_path}: "
                f"bbox={bbox}, image_size=({image_width}, {image_height})"
            )
            return False

        # bbox 크기 확인
        if w < self.min_bbox_size or h < self.min_bbox_size:
            self.warnings.append(
                f"bbox가 너무 작음 {image_path}: {w}x{h} < {self.min_bbox_size}"
            )

        return True

    def validate_annotation(
        self,
        annotation: Dict,
        image_width: int,
        image_height: int,
        required_fields: List[str] = None
    ) -> bool:
        """
        어노테이션 데이터 유효성 검사

        Args:
            annotation: 어노테이션 딕셔너리
            image_width: 이미지 너비
            image_height: 이미지 높이
            required_fields: 필수 필드 리스트

        Returns:
            유효하면 True, 아니면 False
        """
        if required_fields is None:
            required_fields = ['category', 'bbox']

        # 필수 필드 확인
        for field in required_fields:
            if field not in annotation:
                self.errors.append(f"필수 필드 누락: {field}")
                return False

        # bbox 검증
        bbox = annotation.get('bbox', [])
        return self.validate_bbox(bbox, image_width, image_height)

    def validate_dataset(
        self,
        json_data_list: List[Dict],
        image_dir: str
    ) -> Tuple[bool, List[str], List[str]]:
        """
        전체 데이터셋 유효성 검사

        Args:
            json_data_list: JSON 데이터 리스트
            image_dir: 이미지 디렉토리 경로

        Returns:
            (검증 성공 여부, 에러 리스트, 경고 리스트)
        """
        self.errors = []
        self.warnings = []

        for idx, json_data in enumerate(json_data_list):
            # 이미지 정보 추출
            image_info = json_data.get('image', json_data.get('images', {}))
            file_name = image_info.get('file_name', f'unknown_{idx}.jpg')
            image_path = os.path.join(image_dir, file_name)

            # 이미지 검증
            if not self.validate_image(image_path):
                continue

            # 이미지 크기 가져오기
            try:
                img = Image.open(image_path)
                image_width, image_height = img.size
            except:
                self.errors.append(f"이미지 크기 가져오기 실패: {image_path}")
                continue

            # 어노테이션 검증
            annotations = json_data.get('annotations', json_data.get('objects', []))
            for ann in annotations:
                self.validate_annotation(ann, image_width, image_height)

        # 결과 요약
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings

    def print_summary(self):
        """검증 결과 요약 출력"""
        print("\n===== 데이터 검증 결과 =====")
        print(f"에러: {len(self.errors)}개")
        print(f"경고: {len(self.warnings)}개")

        if self.errors:
            print("\n[에러 목록]")
            for err in self.errors[:10]:  # 최대 10개만 출력
                print(f"  - {err}")
            if len(self.errors) > 10:
                print(f"  ... 외 {len(self.errors) - 10}개")

        if self.warnings:
            print("\n[경고 목록]")
            for warn in self.warnings[:10]:  # 최대 10개만 출력
                print(f"  - {warn}")
            if len(self.warnings) > 10:
                print(f"  ... 외 {len(self.warnings) - 10}개")


def validate_dataset(json_data_list: List[Dict], image_dir: str) -> bool:
    """
    간편 검증 함수 (함수형 인터페이스)

    Args:
        json_data_list: JSON 데이터 리스트
        image_dir: 이미지 디렉토리 경로

    Returns:
        검증 성공 여부
    """
    validator = DataValidator()
    is_valid, errors, warnings = validator.validate_dataset(json_data_list, image_dir)
    validator.print_summary()
    return is_valid


if __name__ == "__main__":
    # 테스트 코드
    print("데이터 검증기 테스트")
    print("실제 데이터 다운로드 후 테스트하세요.")