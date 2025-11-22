"""
전처리 파이프라인 카테고리별 테스트
각 카테고리(도크설비, 보관, 부가가치서비스, 설비및장비, 운반)별로 전처리
"""

import sys
from pathlib import Path
import shutil

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.data_loader import load_json_labels, get_image_info
from tqdm import tqdm


# 카테고리 매핑 (라벨 폴더명 -> 이미지 폴더명)
CATEGORIES = {
    "TL_01_도크설비": "TS_01_도크설비",
    "TL_02_보관": "TS_02_보관",
    "TL_03_부가가치서비스": "TS_03_부가가치서비스",
    "TL_04_설비 및 장비": "TS_04_설비 및 장비",
    "TL_05_운반": "TS_05_운반",
}


def process_category(category_label, category_image, output_base, num_samples=5):
    """
    단일 카테고리 전처리

    Args:
        category_label: 라벨 폴더명 (예: TL_01_도크설비)
        category_image: 이미지 폴더명 (예: TS_01_도크설비)
        output_base: 출력 기본 경로
        num_samples: 테스트할 샘플 수
    """
    print(f"\n{'='*70}")
    print(f" 📦 {category_label} 전처리")
    print(f"{'='*70}")

    # 경로 설정
    label_base = project_root / "data" / "traning" / "label" / category_label
    image_base = project_root / "data" / "traning" / "original" / category_image
    output_dir = output_base / category_label

    # 출력 디렉토리 초기화
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # 하위 폴더 탐색 (불안전한 상태, 불안전한 행동, 작업상황 등)
    sub_folders = [f for f in label_base.iterdir() if f.is_dir()]

    if not sub_folders:
        print(f"  ⚠️  하위 폴더가 없습니다: {label_base}")
        return False

    print(f"  하위 폴더: {[f.name for f in sub_folders]}")

    # 임시 디렉토리
    temp_label_dir = project_root / "data" / "temp_labels"
    temp_image_dir = project_root / "data" / "temp_images"

    if temp_label_dir.exists():
        shutil.rmtree(temp_label_dir)
    if temp_image_dir.exists():
        shutil.rmtree(temp_image_dir)

    temp_label_dir.mkdir(parents=True)
    temp_image_dir.mkdir(parents=True)

    # 각 하위 폴더에서 샘플 수집
    total_copied = 0
    for sub_folder in sub_folders:
        # 대응하는 이미지 폴더
        image_sub_folder = image_base / sub_folder.name

        if not image_sub_folder.exists():
            print(f"  ⚠️  이미지 폴더 없음: {image_sub_folder.name}")
            continue

        # JSON 파일 선택
        json_files = list(sub_folder.glob("*.json"))[:num_samples]

        for json_file in json_files:
            try:
                # JSON 복사
                shutil.copy(json_file, temp_label_dir / json_file.name)

                # 대응하는 이미지 찾기
                json_data = load_json_labels(str(json_file))
                image_info = get_image_info(json_data)
                image_path = image_sub_folder / image_info['file_name']

                if image_path.exists():
                    shutil.copy(image_path, temp_image_dir / image_info['file_name'])
                    total_copied += 1
            except Exception as e:
                print(f"  ❌ 오류: {json_file.name} - {e}")

    print(f"  ✅ {total_copied}개 파일 쌍 준비 완료")

    if total_copied == 0:
        print(f"  ⚠️  처리할 파일이 없습니다.")
        shutil.rmtree(temp_label_dir)
        shutil.rmtree(temp_image_dir)
        return False

    # 파이프라인 실행
    pipeline = PreprocessingPipeline(
        input_dir=str(temp_label_dir.parent),
        output_dir=str(output_dir),
        apply_augmentation=True,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        min_bbox_size=10
    )

    # 데이터 분할
    json_list = list(temp_label_dir.glob("*.json"))
    splits = pipeline.split_dataset([str(f) for f in json_list])

    # 각 split별 처리
    for split_name, split_files in splits.items():
        success_count = 0
        for idx, json_path in enumerate(tqdm(split_files, desc=f"  {split_name}")):
            if pipeline.process_single_data(json_path, str(temp_image_dir), split_name, idx):
                success_count += 1

    # 결과 확인
    for split in ['train', 'val', 'test']:
        split_path = output_dir / split / 'images'
        if split_path.exists():
            image_count = len(list(split_path.glob('*.jpg')))
            print(f"  {split:5s}: {image_count}개")

    # 임시 디렉토리 정리
    shutil.rmtree(temp_label_dir)
    shutil.rmtree(temp_image_dir)

    return True


def test_small_sample(num_samples=1):
    """
    각 카테고리별로 작은 샘플 전처리 테스트

    Args:
        num_samples: 각 하위 폴더에서 가져올 샘플 수
    """
    print("\n" + "="*70)
    print(" 전처리 파이프라인 카테고리별 테스트")
    print(f" (각 하위 폴더에서 {num_samples}개씩 샘플링)")
    print("="*70)

    output_base = project_root / "data" / "processed_test"

    # 출력 베이스 디렉토리 초기화
    if output_base.exists():
        print(f"\n기존 출력 디렉토리 삭제: {output_base}")
        shutil.rmtree(output_base)

    output_base.mkdir(parents=True)

    # 각 카테고리 처리
    success_count = 0
    for label_folder, image_folder in CATEGORIES.items():
        if process_category(label_folder, image_folder, output_base, num_samples):
            success_count += 1

    # 최종 결과
    print("\n" + "="*70)
    print(" 최종 결과")
    print("="*70)
    print(f"  처리 완료: {success_count}/{len(CATEGORIES)} 카테고리")
    print(f"  📁 결과 확인: {output_base}")
    print("="*70)


if __name__ == "__main__":
    test_small_sample(num_samples=1)