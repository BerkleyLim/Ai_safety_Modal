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
    if detection_result and detection_result.status == "anomaly_detected":
        analysis_result = analyze_risk_with_vlm(detection_result) # <-- 실제 VLM 호출
        if analysis_result:
            generate_safety_guideline(analysis_result) # <-- 아직 더미 함수
    print("====== 파이프라인 종료 ======\n")

if __name__ == "__main__":

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
            # "../data/mock/mock_3.png" # 파일 이름 확인 (이전에는 mock_3.png 였음)
            "../data/121.물류창고 내 작업 안전 데이터/01.데이터/2.Validation/원천데이터/VS_05_운반/불안전한 상태(UC)/L-211021_G09_B_UC-03_001_0003.jpg",
            "../data/121.물류창고 내 작업 안전 데이터/01.데이터/2.Validation/원천데이터/VS_05_운반/불안전한 행동(UA)/L-210806_B02_B_UA-01_001_0101.jpg"
        ]

        # 4. 메인 파이프라인 실행
        for image_path in image_files:
            if not os.path.exists(image_path):
                print(f"오류: 테스트 이미지 파일이 없습니다. '{image_path}' 경로를 확인해주세요.")
            else:
                main_pipeline(image_path)