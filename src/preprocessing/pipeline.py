"""
전체 전처리 파이프라인
데이터 로딩 → 검증 → 증강 → 저장까지 통합 처리합니다.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from tqdm import tqdm
import shutil

from .data_loader import load_json_labels, load_image, scan_dataset, get_image_info, parse_annotations
from .data_validator import DataValidator
from .data_augmentation import ImageAugmenter


class PreprocessingPipeline:
    """전처리 파이프라인"""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        apply_augmentation: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        min_bbox_size: int = 10,
    ):
        """
        Args:
            input_dir: 원본 데이터 디렉토리
            output_dir: 전처리 결과 저장 디렉토리
            apply_augmentation: 데이터 증강 적용 여부
            train_ratio: 학습 데이터 비율
            val_ratio: 검증 데이터 비율
            test_ratio: 테스트 데이터 비율
            min_bbox_size: 최소 bbox 크기 (픽셀)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.apply_augmentation = apply_augmentation
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # 검증기 및 증강기 초기화
        self.validator = DataValidator(min_bbox_size=min_bbox_size)
        self.augmenter = ImageAugmenter() if apply_augmentation else None

        # 출력 디렉토리 생성
        self._create_output_dirs()

    def _create_output_dirs(self):
        """출력 디렉토리 구조 생성"""
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        print(f"출력 디렉토리 생성 완료: {self.output_dir}")

    def split_dataset(
        self,
        file_list: List[str]
    ) -> Dict[str, List[str]]:
        """
        데이터셋을 train/val/test로 분할합니다.

        Args:
            file_list: 파일 경로 리스트

        Returns:
            {'train': [...], 'val': [...], 'test': [...]}
        """
        import random
        random.shuffle(file_list)

        total = len(file_list)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        splits = {
            'train': file_list[:train_end],
            'val': file_list[train_end:val_end],
            'test': file_list[val_end:]
        }

        print(f"\n데이터 분할 완료:")
        print(f"  - Train: {len(splits['train'])}개")
        print(f"  - Val: {len(splits['val'])}개")
        print(f"  - Test: {len(splits['test'])}개")

        return splits

    def process_single_data(
        self,
        json_path: str,
        image_dir: str,
        split: str,
        index: int
    ) -> bool:
        """
        단일 데이터를 처리합니다.

        Args:
            json_path: JSON 라벨 파일 경로
            image_dir: 이미지 디렉토리
            split: 'train', 'val', 'test' 중 하나
            index: 데이터 인덱스 (파일명에 사용)

        Returns:
            성공 여부
        """
        try:
            # JSON 로드
            json_data = load_json_labels(json_path)

            # 이미지 정보 추출
            image_info = get_image_info(json_data)
            image_filename = image_info.get('file_name', 'unknown.jpg')
            image_path = os.path.join(image_dir, image_filename)

            # 이미지 로드
            if not self.validator.validate_image(image_path):
                return False

            image = load_image(image_path, mode='PIL')
            width, height = image.size

            # 어노테이션 추출
            annotations = parse_annotations(json_data)

            # bbox 추출
            bboxes = [ann['bbox'] for ann in annotations]

            # bbox 검증
            valid_bboxes = []
            valid_annotations = []
            for bbox, ann in zip(bboxes, annotations):
                if self.validator.validate_bbox(bbox, width, height, image_path):
                    valid_bboxes.append(bbox)
                    valid_annotations.append(ann)

            if not valid_bboxes:
                print(f"  ⚠️  유효한 bbox가 없음: {json_path}")
                return False

            # 증강 적용
            if self.apply_augmentation and split == 'train':
                augmented_results = self.augmenter.augment(
                    image, valid_bboxes, augment_all=False
                )
            else:
                augmented_results = [(image, valid_bboxes, "original")]

            # 증강된 데이터 저장
            for aug_idx, (aug_img, aug_bboxes, aug_method) in enumerate(augmented_results):
                # 파일명 생성
                output_filename = f"{split}_{index:06d}_{aug_idx}"
                output_image_path = self.output_dir / split / 'images' / f"{output_filename}.jpg"
                output_label_path = self.output_dir / split / 'labels' / f"{output_filename}.json"

                # 이미지 저장
                aug_img.save(output_image_path, quality=95)

                # 라벨 저장
                output_label = {
                    'image': {
                        'file_name': f"{output_filename}.jpg",
                        'width': aug_img.width,
                        'height': aug_img.height,
                    },
                    'annotations': [
                        {
                            'category': ann['category'],
                            'bbox': bbox,
                            'attributes': ann.get('attributes', {})
                        }
                        for bbox, ann in zip(aug_bboxes, valid_annotations[:len(aug_bboxes)])
                    ],
                    'augmentation': aug_method,
                    'source_file': str(json_path)
                }

                with open(output_label_path, 'w', encoding='utf-8') as f:
                    json.dump(output_label, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"  ❌ 처리 실패 {json_path}: {str(e)}")
            return False

    def run(self, image_subdir: str = "images", label_subdir: str = "labels"):
        """
        전체 파이프라인을 실행합니다.

        Args:
            image_subdir: 이미지가 있는 하위 디렉토리명
            label_subdir: 라벨이 있는 하위 디렉토리명
        """
        print("\n" + "="*60)
        print("데이터 전처리 파이프라인 시작")
        print("="*60)

        # JSON 파일 스캔
        print(f"\n1. JSON 라벨 파일 스캔 중...")
        label_dir = self.input_dir / label_subdir
        json_files = scan_dataset(str(label_dir), ext='.json')
        print(f"   발견된 JSON 파일: {len(json_files)}개")

        if not json_files:
            print("❌ JSON 파일을 찾을 수 없습니다. 경로를 확인하세요.")
            return

        # 데이터 분할
        print(f"\n2. 데이터 분할 중...")
        splits = self.split_dataset(json_files)

        # 각 split별 처리
        image_dir = self.input_dir / image_subdir

        for split_name, split_files in splits.items():
            print(f"\n3. {split_name.upper()} 데이터 처리 중...")

            success_count = 0
            for idx, json_path in enumerate(tqdm(split_files, desc=f"  Processing {split_name}")):
                if self.process_single_data(json_path, str(image_dir), split_name, idx):
                    success_count += 1

            print(f"   ✅ {split_name}: {success_count}/{len(split_files)} 성공")

        # 검증 결과 출력
        print("\n4. 검증 결과:")
        self.validator.print_summary()

        print("\n" + "="*60)
        print("전처리 파이프라인 완료")
        print(f"결과 저장 위치: {self.output_dir}")
        print("="*60)


def preprocess_pipeline(
    input_dir: str,
    output_dir: str,
    apply_augmentation: bool = True,
    **kwargs
):
    """
    간편 전처리 함수 (함수형 인터페이스)

    Args:
        input_dir: 원본 데이터 디렉토리
        output_dir: 전처리 결과 저장 디렉토리
        apply_augmentation: 데이터 증강 적용 여부
        **kwargs: PreprocessingPipeline 추가 파라미터
    """
    pipeline = PreprocessingPipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        apply_augmentation=apply_augmentation,
        **kwargs
    )
    pipeline.run()


if __name__ == "__main__":
    # 테스트 코드
    print("전처리 파이프라인 테스트")
    print("실제 사용 예시:")
    print("""
    from preprocessing import preprocess_pipeline

    preprocess_pipeline(
        input_dir='../data/raw',
        output_dir='../data/processed',
        apply_augmentation=True,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    """)