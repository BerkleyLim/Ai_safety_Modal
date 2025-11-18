"""
전처리 파이프라인 사용 예시
실제 데이터 다운로드 후 이 스크립트를 실행하세요.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from preprocessing.pipeline import preprocess_pipeline


def main():
    """전처리 파이프라인 실행"""

    # ========================================
    # 설정값 (실제 데이터에 맞게 수정하세요)
    # ========================================

    # 원본 데이터 경로
    INPUT_DIR = project_root / "data" / "raw"

    # 전처리 결과 저장 경로
    OUTPUT_DIR = project_root / "data" / "processed"

    # 전처리 옵션
    APPLY_AUGMENTATION = True  # 데이터 증강 적용 여부
    TRAIN_RATIO = 0.7          # 학습 데이터 비율 (70%)
    VAL_RATIO = 0.15           # 검증 데이터 비율 (15%)
    TEST_RATIO = 0.15          # 테스트 데이터 비율 (15%)
    MIN_BBOX_SIZE = 10         # 최소 bbox 크기 (픽셀)

    # ========================================
    # 전처리 실행
    # ========================================

    print("\n" + "="*70)
    print(" 물류창고 안전 데이터셋 전처리 시작")
    print("="*70)
    print(f"\n입력 디렉토리: {INPUT_DIR}")
    print(f"출력 디렉토리: {OUTPUT_DIR}")
    print(f"데이터 증강: {'적용' if APPLY_AUGMENTATION else '미적용'}")
    print(f"데이터 분할 비율: Train {TRAIN_RATIO*100:.0f}% / Val {VAL_RATIO*100:.0f}% / Test {TEST_RATIO*100:.0f}%")

    # 입력 디렉토리 확인
    if not INPUT_DIR.exists():
        print(f"\n❌ 오류: 입력 디렉토리가 존재하지 않습니다: {INPUT_DIR}")
        print("데이터를 다운로드하고 올바른 경로를 설정하세요.")
        return

    # 파이프라인 실행
    preprocess_pipeline(
        input_dir=str(INPUT_DIR),
        output_dir=str(OUTPUT_DIR),
        apply_augmentation=APPLY_AUGMENTATION,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        min_bbox_size=MIN_BBOX_SIZE
    )

    print("\n✅ 전처리 완료!")
    print(f"결과 확인: {OUTPUT_DIR}")


def test_single_file():
    """단일 파일 테스트용 함수"""

    from preprocessing.data_loader import load_json_labels, load_image, parse_annotations
    from preprocessing.data_validator import DataValidator
    from preprocessing.data_augmentation import ImageAugmenter

    print("\n" + "="*70)
    print(" 단일 파일 테스트")
    print("="*70)

    # 테스트할 파일 경로 (실제 파일로 변경하세요)
    json_path = project_root / "data" / "raw" / "labels" / "sample_001.json"
    image_dir = project_root / "data" / "raw" / "images"

    if not json_path.exists():
        print(f"\n⚠️  테스트 파일이 없습니다: {json_path}")
        print("실제 데이터 다운로드 후 경로를 수정하세요.")
        return

    # JSON 로드
    print(f"\n1. JSON 로드: {json_path}")
    json_data = load_json_labels(str(json_path))
    print(f"   ✅ 로드 완료")

    # 어노테이션 파싱
    print(f"\n2. 어노테이션 파싱")
    annotations = parse_annotations(json_data)
    print(f"   객체 수: {len(annotations)}")
    for i, ann in enumerate(annotations[:3]):  # 최대 3개만 출력
        print(f"   - 객체 {i+1}: {ann['category']}, bbox: {ann['bbox']}")

    # 이미지 로드
    image_info = json_data.get('image', json_data.get('images', {}))
    image_filename = image_info.get('file_name', 'unknown.jpg')
    image_path = image_dir / image_filename

    print(f"\n3. 이미지 로드: {image_path}")
    if not image_path.exists():
        print(f"   ❌ 이미지 파일이 없습니다: {image_path}")
        return

    image = load_image(str(image_path), mode='PIL')
    print(f"   ✅ 이미지 크기: {image.size}")

    # 검증
    print(f"\n4. 데이터 검증")
    validator = DataValidator(min_bbox_size=10)
    is_valid = validator.validate_image(str(image_path))
    print(f"   이미지 유효성: {'✅ 통과' if is_valid else '❌ 실패'}")

    bboxes = [ann['bbox'] for ann in annotations]
    for i, bbox in enumerate(bboxes):
        is_valid = validator.validate_bbox(bbox, image.width, image.height)
        status = '✅' if is_valid else '❌'
        print(f"   {status} bbox {i+1}: {bbox}")

    # 증강 테스트
    print(f"\n5. 데이터 증강 테스트")
    augmenter = ImageAugmenter()
    augmented = augmenter.augment(image, bboxes, augment_all=False)
    print(f"   생성된 증강 이미지: {len(augmented)}개")
    for img, aug_bboxes, method in augmented:
        print(f"   - {method}: {len(aug_bboxes)} objects, size: {img.size}")

    print("\n✅ 단일 파일 테스트 완료!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='데이터 전처리 파이프라인')
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'test'],
        help='실행 모드: full (전체 전처리) 또는 test (단일 파일 테스트)'
    )

    args = parser.parse_args()

    if args.mode == 'full':
        main()
    elif args.mode == 'test':
        test_single_file()