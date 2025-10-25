# src/action.py

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일 로드 및 OpenAI 클라이언트 초기화 (reasoning.py와 동일하게 설정)
load_dotenv()
api_key_from_env = os.environ.get("OPENAI_API_KEY")
if api_key_from_env:
    client = OpenAI(api_key=api_key_from_env)
else:
    print("🚨 [Action] 오류: OPENAI_API_KEY 환경 변수를 찾을 수 없습니다.")
    client = None

def generate_safety_guideline(analysis_result):
    """
    [실제 Action Layer 함수]
    VLM 분석 결과를 바탕으로 위험 등급에 따라 조치를 취하고,
    HIGH 위험 시 LLM을 호출하여 다국어 안전 지침을 생성합니다.
    """
    if not client:
        print("🚫 [Action] OpenAI 클라이언트가 초기화되지 않아 LLM 호출을 건너<0xEB><0><0xA4>니다.")
        return {"status": "error_client_not_initialized"}

    risk_level = analysis_result.get("risk_level", "Unknown")
    reason = analysis_result.get("reason", "제공된 이유 없음.")

    print(f"📢 [Action] 위험 등급 '{risk_level}'에 따른 조치 실행...")

    if risk_level == "LOW":
        print("➡️ [Action] 위험 등급 LOW: 로그 기록만 수행합니다.")
        # 여기에 로그 기록 로직 추가 (예: 파일 저장, DB 저장 등)
        return {"status": "logged"}

    elif risk_level == "MED":
        print("⚠️ [Action] 위험 등급 MED: 확인 알림 표시.")
        print(f"   - 확인 필요: {reason}")
        # 여기에 확인 알림 UI 표시 또는 메시지 전송 로직 추가
        return {"status": "confirmation_requested"}

    elif risk_level == "HIGH":
        print("🚨 [Action] 위험 등급 HIGH: LLM 호출하여 다국어 안전 지침 생성...")

        # LLM에게 전달할 프롬프트 구성
        prompt_for_llm = f"""
        다음은 스마트 공장에서 감지된 심각한 안전 위험 상황입니다:
        상황 설명: {reason}

        이 상황에 대한 구체적인 행동 지침을 생성해주세요.
        지침은 다음 언어로 각각 작성해야 합니다: 한국어, 영어, 베트남어.

        결과는 아래와 같은 JSON 형식으로만 응답해주세요:
        {{
          "guideline_ko": "...",
          "guideline_en": "...",
          "guideline_vi": "..."
        }}
        """

        try:
            # LLM API 호출 (텍스트 생성 모델 사용, 예: gpt-4o 또는 gpt-3.5-turbo)
            response = client.chat.completions.create(
                model="gpt-4o", # 또는 비용 효율적인 모델 선택
                messages=[
                    {"role": "user", "content": prompt_for_llm}
                ],
                max_tokens=500, # 충분한 길이의 지침 생성 허용
                response_format={"type": "json_object"} # JSON 응답 강제
            )

            guideline_str = response.choices[0].message.content
            guidelines = json.loads(guideline_str)

            print("--- [생성된 다국어 안전 지침] ---")
            print(f"🇰🇷 (한국어): {guidelines.get('guideline_ko', '생성 실패')}")
            print(f"🇺🇸 (English): {guidelines.get('guideline_en', 'Generation failed')}")
            print(f"🇻🇳 (Tiếng Việt): {guidelines.get('guideline_vi', 'Tạo không thành công')}")
            print("---------------------------------")
            # 여기에 생성된 지침을 실제 사용자에게 전달하는 로직 추가 (예: 앱 푸시, SMS 등)
            return {"status": "multilingual_guideline_generated", "guidelines": guidelines}

        except Exception as e:
            print(f"🚨 [Action] LLM API 호출 중 오류 발생: {e}")
            return {"status": "error_llm_api_call"}

    else:
        print(f"❓ [Action] 알 수 없는 위험 등급: {risk_level}")
        return {"status": "unknown_risk_level"}