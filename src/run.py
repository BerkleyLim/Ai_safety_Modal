# src/run.py

import os
import sys
import glob
import random
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
# --- 모듈 임포트 ---
from monitoring import detect_objects
from reasoning import analyze_risk_with_vlm
from action import generate_safety_guideline

# 1. 로그 저장용 클래스 정의 (화면 + 파일 동시 출력)
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message) # 화면에 출력
        self.log.write(message)      # 파일에 저장

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 1. 환경 변수 로드
load_dotenv()

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

def main_pipeline(image_path):
    """전체 안전 관제 파이프라인을 실행하는 메인 함수"""
    from schemas.monitoring_output import MonitoringOutput
    print(f"\n====== [TEST] 파일: {os.path.basename(image_path)} ======")
    
    # 1. Monitoring Layer 실행
    detection_result = detect_objects(image_path)
    
    # [수정] Pydantic 모델은 딕셔너리 접근['key']이 아니라 속성 접근(.key)을 해야 합니다.
    if detection_result and detection_result.status == "anomaly_detected":

        # 2. Reasoning Layer 실행
        analysis_result = analyze_risk_with_vlm(detection_result)
        
        # 3. Action Layer 실행
        if analysis_result:
            action_output = generate_safety_guideline(analysis_result)
            
            # 결과 출력
            print(f"🎬 [Pipeline] 최종 조치 상태: {action_output.status}")
            if action_output.guidelines:
                print(f"📝 [Pipeline] 생성된 지침(KO - 요약): {action_output.guidelines.guideline_ko[:100]}...")
    else:
        print("✅ [Pipeline] 정상/무시됨 (VLM 호출 안 함)")

    print("=======================================================\n")


if __name__ == "__main__":
# 로그 폴더 생성
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 파일명 생성 (예: logs/run_20251129_123000.txt)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{timestamp}.txt")
    
    # [핵심] 표준 출력(stdout)을 DualLogger로 교체
    # 이제부터 모든 파일의 print()는 이 클래스를 통과합니다.
    sys.stdout = DualLogger(log_path)
    
    print(f"📝 로그가 저장됩니다: {log_path}")

    import argparse

    parser = argparse.ArgumentParser(description='물류창고 안전 관제 시스템 실행')
    
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='분석할 이미지 경로 (예: ../data/01_도크설비/logistics_yolo/val/images/image_000001.jpg)'
    )
    
    # [추가] 외장하드 등 데이터 루트 경로 지정
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help='데이터셋 루트 경로 (예: /Volumes/Elements/data)'
    )

    args = parser.parse_args()

    # API 키 확인
    if not os.environ.get("OPENAI_API_KEY"):
        print("🚨 오류: OPENAI_API_KEY가 .env 파일에 없습니다.")
        print("프로젝트 루트에 .env 파일을 만들고 키를 입력하세요.")
        sys.exit(1)
    else:
        print("🔑 OpenAI API Key 로드 완료.")

    image_files = []

    # 1. 사용자가 특정 이미지를 지정한 경우
    if args.image:
        if os.path.exists(args.image):
            image_files = [args.image]
        else:
            print(f"🚨 오류: 지정한 이미지 파일을 찾을 수 없습니다: {args.image}")
            sys.exit(1)

    # 2. 데이터 루트(외장하드)를 지정한 경우 -> 자동으로 검증 이미지 찾기
    elif args.data_root:
        print(f"📂 데이터 루트에서 검증 이미지 검색 중: {args.data_root}")
        # 외장하드 구조: [Root]/[카테고리]/logistics_yolo/val/images/*.jpg
        search_pattern = os.path.join(args.data_root, "**", "logistics_yolo", "val", "images", "*.[jp][pn]g")
        found_images = glob.glob(search_pattern, recursive=True)
        
        if found_images:
            # 너무 많으면 3개만 랜덤 선택
            sample_count = min(10, len(found_images))
            image_files = random.sample(found_images, sample_count)
            print(f"👉 총 {len(found_images)}장 중 {sample_count}장을 무작위로 선택했습니다.")
        else:
            print("🚨 오류: 해당 경로에서 'logistics_yolo/val/images' 내의 이미지를 찾을 수 없습니다.")
            sys.exit(1)

    # 3. 아무것도 지정 안 함 -> 기본 mock 데이터 사용
    else:
        print("👉 별도 경로 지정 없음.")
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
        main_pipeline(img_path)