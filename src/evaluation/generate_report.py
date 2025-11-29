# src/evaluation/generate_report.py
"""학습 결과 리포트 및 시각화 생성 스크립트"""

import argparse
import sys
from pathlib import Path

# matplotlib 백엔드 설정 (GUI 없이 저장)
import matplotlib
matplotlib.use('Agg')

from .metrics import print_model_summary, find_model_results
from .validation import run_full_validation
from .visualize import visualize_model_results


def generate_all_reports(models_dir: str, output_dir: str = None):
    """모든 학습된 모델에 대해 리포트 생성"""
    models_path = Path(models_dir)
    model_dirs = sorted(models_path.glob("safety_*"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not model_dirs:
        print("❌ 학습된 모델이 없습니다.")
        return

    print(f"📁 {len(model_dirs)}개 모델 발견\n")

    for model_dir in model_dirs:
        print(f"\n{'='*60}")
        print(f"📊 {model_dir.name}")
        print('='*60)

        # 시각화 생성
        visualize_model_results(str(model_dir), output_dir)


def main():
    parser = argparse.ArgumentParser(description='학습 결과 리포트 및 시각화 생성')
    parser.add_argument(
        '--action',
        type=str,
        choices=['metrics', 'validate', 'visualize', 'all'],
        default='all',
        help='실행할 작업 (metrics: 성능 지표, validate: 적합성 검증, visualize: 시각화, all: 전체)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='특정 모델 폴더 경로 (미지정 시 가장 최근 모델)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='시각화 저장 경로 (미지정 시 모델 폴더에 저장)'
    )

    args = parser.parse_args()

    # 프로젝트 경로
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"

    # 특정 모델 또는 최신 모델 선택
    if args.model:
        model_dir = Path(args.model)
    else:
        model_dirs = sorted(models_dir.glob("safety_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not model_dirs:
            print("❌ 학습된 모델이 없습니다.")
            sys.exit(1)
        model_dir = model_dirs[0]

    print(f"🎯 대상 모델: {model_dir.name}\n")

    # 작업 실행
    if args.action in ['metrics', 'all']:
        print_model_summary(str(models_dir))

    if args.action in ['validate', 'all']:
        # 테스트 이미지 찾기
        test_images = list((project_root / "data").rglob("**/val/images/*.jpg"))
        test_image = str(test_images[0]) if test_images else None

        validation = run_full_validation(str(model_dir), test_image)
        validation.print_summary()

    if args.action in ['visualize', 'all']:
        visualize_model_results(str(model_dir), args.output)
        print(f"\n✅ 시각화 저장 완료: {model_dir}/training_dashboard.png")


if __name__ == "__main__":
    main()