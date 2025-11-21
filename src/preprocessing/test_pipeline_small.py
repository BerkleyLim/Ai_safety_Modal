"""
전처리 파이프라인 작은 샘플 테스트
5개 데이터만 처리해서 파이프라인 동작 확인
"""

import sys
from pathlib import Path
import shutil

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from preprocessing.pipeline import PreprocessingPipeline


def test_small_sample():
    """작은 샘플로 전처리 파이프라인 테스트"""

    print("\n" + "="*70)
    print(" 전처리 파이프라인 작은 샘플 테스트 (5개)")
    print("="*70)

    # 경로 설정
    LABEL_DIR = project_root / "data" / "traning" / "label" / "TL_01_도크설비" / "불안전한 상태(UC)"
    IMAGE_DIR = project_root / "data" / "traning" / "original" / "TS_01_도크설비" / "불안전한 상태(UC)"
    OUTPUT_DIR = project_root / "data" / "processed_test"

    # 출력 디렉토리 초기화
    if OUTPUT_DIR.exists():
        print(f"\n기존 테스트 출력 디렉토리 삭제: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    print(f"\n📂 입력:")
    print(f"  라벨: {LABEL_DIR}")
    print(f"  이미지: {IMAGE_DIR}")
    print(f"\n📂 출력:")
    print(f"  {OUTPUT_DIR}")

    # JSON 파일 5개만 선택
    json_files = list(LABEL_DIR.glob("*.json"))[:5]

    if not json_files:
        print(f"\n❌ JSON 파일을 찾을 수 없습니다: {LABEL_DIR}")
        return

    print(f"\n✅ 테스트할 JSON 파일 {len(json_files)}개:")
    for jf in json_files:
        print(f"  - {jf.name}")

    # 임시 디렉토리 생성
    temp_label_dir = project_root / "data" / "temp_test_labels"
    temp_image_dir = project_root / "data" / "temp_test_images"

    # 기존 임시 디렉토리 정리
    if temp_label_dir.exists():
        shutil.rmtree(temp_label_dir)
    if temp_image_dir.exists():
        shutil.rmtree(temp_image_dir)

    temp_label_dir.mkdir(parents=True)
    temp_image_dir.mkdir(parents=True)

    # JSON과 대응하는 이미지 복사
    print(f"\n📋 파일 준비 중...")
    from preprocessing.data_loader import load_json_labels, get_image_info

    copied_count = 0
    for json_file in json_files:
        # JSON 복사
        shutil.copy(json_file, temp_label_dir / json_file.name)

        # 대응하는 이미지 찾기
        json_data = load_json_labels(str(json_file))
        image_info = get_image_info(json_data)
        image_path = IMAGE_DIR / image_info['file_name']

        if image_path.exists():
            shutil.copy(image_path, temp_image_dir / image_info['file_name'])
            copied_count += 1
            print(f"  ✅ {json_file.name} + {image_info['file_name']}")

    print(f"\n✅ {copied_count}개 파일 쌍 준비 완료")

    # 파이프라인 실행
    print(f"\n🔧 전처리 파이프라인 실행...")

    pipeline = PreprocessingPipeline(
        input_dir=str(temp_label_dir.parent),  # labels의 부모 디렉토리
        output_dir=str(OUTPUT_DIR),
        apply_augmentation=True,
        train_ratio=0.6,  # 5개 중 3개
        val_ratio=0.2,    # 5개 중 1개
        test_ratio=0.2,   # 5개 중 1개
        min_bbox_size=10
    )

    # 실행 (image_subdir, label_subdir 수정 필요)
    # 임시로 직접 처리
    import json
    from tqdm import tqdm

    print(f"\n1. 데이터 분할 중...")
    json_list = list(temp_label_dir.glob("*.json"))
    splits = pipeline.split_dataset([str(f) for f in json_list])

    print(f"\n2. 각 split별 처리 중...")
    for split_name, split_files in splits.items():
        print(f"\n  처리 중: {split_name.upper()}")
        success_count = 0

        for idx, json_path in enumerate(tqdm(split_files, desc=f"    {split_name}")):
            if pipeline.process_single_data(json_path, str(temp_image_dir), split_name, idx):
                success_count += 1

        print(f"    ✅ {split_name}: {success_count}/{len(split_files)} 성공")

    print(f"\n3. 검증 결과:")
    pipeline.validator.print_summary()

    # 결과 확인
    print(f"\n" + "="*70)
    print(" 결과 확인")
    print("="*70)

    for split in ['train', 'val', 'test']:
        image_count = len(list((OUTPUT_DIR / split / 'images').glob('*.jpg')))
        label_count = len(list((OUTPUT_DIR / split / 'labels').glob('*.json')))
        print(f"  {split:5s}: 이미지 {image_count}개, 라벨 {label_count}개")

    # 임시 디렉토리 정리
    print(f"\n🧹 임시 파일 정리...")
    shutil.rmtree(temp_label_dir)
    shutil.rmtree(temp_image_dir)

    print(f"\n✅ 테스트 완료!")
    print(f"📁 결과 확인: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    test_small_sample()