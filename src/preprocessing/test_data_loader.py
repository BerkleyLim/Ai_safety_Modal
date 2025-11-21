"""
데이터 로더 테스트 스크립트
실제 데이터로 JSON 파싱과 이미지-라벨 매칭을 테스트합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from preprocessing.data_loader import load_json_labels, load_image, parse_annotations, get_image_info


def test_single_json_and_image():
    """단일 JSON과 이미지 매칭 테스트"""

    print("\n" + "="*70)
    print(" 데이터 로더 테스트")
    print("="*70)

    # 테스트할 경로
    label_dir = project_root / "data" / "traning" / "label" / "TL_01_도크설비" / "불안전한 상태(UC)"
    image_dir = project_root / "data" / "traning" / "original" / "TS_01_도크설비" / "불안전한 상태(UC)"

    # 첫 번째 JSON 파일 찾기
    json_files = list(label_dir.glob("*.json"))

    if not json_files:
        print(f"\n❌ JSON 파일을 찾을 수 없습니다: {label_dir}")
        return

    json_path = json_files[0]
    print(f"\n1️⃣  테스트 JSON: {json_path.name}")

    # JSON 로드
    try:
        json_data = load_json_labels(str(json_path))
        print("   ✅ JSON 로드 성공")
    except Exception as e:
        print(f"   ❌ JSON 로드 실패: {e}")
        return

    # 이미지 정보 추출
    print(f"\n2️⃣  이미지 정보 추출")
    image_info = get_image_info(json_data)
    print(f"   파일명: {image_info['file_name']}")
    print(f"   해상도: {image_info['width']}x{image_info['height']}")
    print(f"   source_data_ID: {image_info['source_data_ID']}")

    # 이미지 파일 경로 생성
    image_path = image_dir / image_info['file_name']
    print(f"\n3️⃣  이미지 로드 시도")
    print(f"   경로: {image_path}")

    # 이미지 로드
    if not image_path.exists():
        print(f"   ❌ 이미지 파일이 없습니다: {image_path}")
        print(f"\n   💡 디렉토리 확인:")
        print(f"      라벨: {label_dir.exists()} - {label_dir}")
        print(f"      이미지: {image_dir.exists()} - {image_dir}")
        return

    try:
        image = load_image(str(image_path), mode='PIL')
        print(f"   ✅ 이미지 로드 성공")
        print(f"   이미지 크기: {image.size}")
    except Exception as e:
        print(f"   ❌ 이미지 로드 실패: {e}")
        return

    # 어노테이션 추출
    print(f"\n4️⃣  어노테이션 파싱")
    annotations = parse_annotations(json_data)
    print(f"   객체 수: {len(annotations)}개")

    if annotations:
        print(f"\n   📦 객체 정보:")
        for i, ann in enumerate(annotations[:5]):  # 최대 5개만 출력
            bbox = ann['bbox']
            print(f"      {i+1}. {ann['category']}")
            print(f"         - bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            print(f"         - type: {ann['type']}")

    print("\n" + "="*70)
    print(" ✅ 테스트 완료!")
    print("="*70)

    return {
        'json_path': json_path,
        'image_path': image_path,
        'image_info': image_info,
        'annotations': annotations
    }


def test_multiple_samples(num_samples=5):
    """여러 샘플을 테스트"""

    print("\n" + "="*70)
    print(f" 다중 샘플 테스트 (최대 {num_samples}개)")
    print("="*70)

    label_dir = project_root / "data" / "traning" / "label" / "TL_01_도크설비" / "불안전한 상태(UC)"
    image_dir = project_root / "data" / "traning" / "original" / "TS_01_도크설비" / "불안전한 상태(UC)"

    json_files = list(label_dir.glob("*.json"))[:num_samples]

    success_count = 0
    fail_count = 0

    for json_path in json_files:
        try:
            json_data = load_json_labels(str(json_path))
            image_info = get_image_info(json_data)
            image_path = image_dir / image_info['file_name']

            if image_path.exists():
                image = load_image(str(image_path), mode='PIL')
                annotations = parse_annotations(json_data)
                print(f"✅ {json_path.name}: {len(annotations)}개 객체")
                success_count += 1
            else:
                print(f"❌ {json_path.name}: 이미지 없음")
                fail_count += 1

        except Exception as e:
            print(f"❌ {json_path.name}: {e}")
            fail_count += 1

    print(f"\n결과: {success_count}개 성공, {fail_count}개 실패")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='데이터 로더 테스트')
    parser.add_argument(
        '--mode',
        type=str,
        default='single',
        choices=['single', 'multiple'],
        help='테스트 모드: single (단일) 또는 multiple (다중)'
    )
    parser.add_argument(
        '--num',
        type=int,
        default=5,
        help='다중 테스트 시 샘플 수'
    )

    args = parser.parse_args()

    if args.mode == 'single':
        test_single_json_and_image()
    elif args.mode == 'multiple':
        test_multiple_samples(num_samples=args.num)