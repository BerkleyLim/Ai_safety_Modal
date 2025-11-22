# src/run.py

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent


def main_pipeline(image_path):
    """전체 안전 관제 파이프라인을 실행하는 메인 함수"""
    from monitoring import detect_objects
    from reasoning import analyze_risk_with_vlm
    from action import generate_safety_guideline

    print("\n====== 전체 안전 관제 파이프라인 시작 ======")
    detection_result = detect_objects(image_path)
    if detection_result and detection_result["status"] == "anomaly_detected":
        analysis_result = analyze_risk_with_vlm(detection_result)
        if analysis_result:
            generate_safety_guideline(analysis_result)
    print("====== 파이프라인 종료 ======\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='물류창고 안전 관제 시스템')
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='분석할 이미지 경로 (예: ../data/01_도크설비/logistics_yolo/val/images/image_000001.jpg)'
    )

    args = parser.parse_args()

    # API 키 확인
    if not os.environ.get("OPENAI_API_KEY"):
        print("오류: OPENAI_API_KEY가 .env 파일에 없습니다.")
        print("프로젝트 루트에 .env 파일을 만들고 키를 입력하세요.")
        sys.exit(1)
    else:
        print("OpenAI API Key 로드 완료.")

    # 이미지 경로 설정
    if args.image:
        image_files = [args.image]
    else:
        # 전처리된 데이터에서 테스트 이미지 찾기
        val_dirs = list((PROJECT_ROOT / "data").glob("**/logistics_yolo/val/images"))
        if val_dirs:
            test_images = list(val_dirs[0].glob("*.jpg")) + list(val_dirs[0].glob("*.png"))
            if test_images:
                image_files = [str(test_images[0])]
                print(f"테스트 이미지 자동 선택: {image_files[0]}")
            else:
                image_files = [str(PROJECT_ROOT / "data" / "mock" / "mock_3.png")]
        else:
            image_files = [str(PROJECT_ROOT / "data" / "mock" / "mock_3.png")]

    # 파이프라인 실행
    for img_path in image_files:
        if not os.path.exists(img_path):
            print(f"오류: 이미지 파일이 없습니다: {img_path}")
        else:
            main_pipeline(img_path)