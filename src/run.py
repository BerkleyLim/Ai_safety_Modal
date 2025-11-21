# src/run.py

from monitoring import detect_objects
from reasoning import analyze_risk_with_vlm # <-- 이제 진짜 VLM 함수를 불러올 겁니다.
from action import generate_safety_guideline
import os
from dotenv import load_dotenv # <-- .env 파일 로드를 위해 추가

# 1. .env 파일에서 환경 변수 로드 (파일 최상단에서 한 번만 해도 됩니다)
load_dotenv()

def main_pipeline(image_path):
    """ 전체 안전 관제 파이프라인을 실행하는 메인 함수 """
    print("\n====== 전체 안전 관제 파이프라인 시작 ======")
    detection_result = detect_objects(image_path)
    if detection_result and detection_result["status"] == "anomaly_detected":
        analysis_result = analyze_risk_with_vlm(detection_result) # <-- 실제 VLM 호출
        if analysis_result:
            generate_safety_guideline(analysis_result) # <-- 아직 더미 함수
    print("====== 파이프라인 종료 ======\n")

if __name__ == "__main__":

    # ========================================
    # 전처리 테스트 (원하는 것만 주석 해제)
    # ========================================

    # 1) 데이터 로더 테스트 (JSON + 이미지 매칭 확인)
    # from preprocessing.test_data_loader import test_single_json_and_image
    # test_single_json_and_image()

    # 2) 작은 샘플 전처리 테스트 (5개 데이터로 전처리 파이프라인 확인)
    from preprocessing.test_pipeline_small import test_small_sample
    test_small_sample()

    # ========================================
    # 메인 파이프라인
    # ========================================

    # 2. API 키가 로드되었는지 확인 (선택 사항이지만 안전)
    if not os.environ.get("OPENAI_API_KEY"):
        print("🚨 오류: OPENAI_API_KEY가 .env 파일에 없거나 로드되지 않았습니다.")
        print("프로젝트 최상위 폴더에 .env 파일을 만들고 키를 입력하세요.")
    else:
        print("🔑 OpenAI API Key 로드 완료.")

        # --- 이전에 있던 OpenAI 테스트 코드는 삭제 ---

        # 3. 테스트할 이미지 파일 목록 정의
        image_files = [
            # "../data/mock/mock_1.png",
            # "../data/mock/mock_2.png",
            "../data/mock/mock_3.png"  # 파일 이름 확인 (이전에는 mock_3.png 였음)
            # 나중에 실제 데이터 추가:
            # "../data/Training/02.라벨링데이터/fire/..."
        ]

        # 4. 메인 파이프라인 실행
        for image_path in image_files:
            if not os.path.exists(image_path):
                print(f"오류: 테스트 이미지 파일이 없습니다. '{image_path}' 경로를 확인해주세요.")
            else:
                main_pipeline(image_path)