# src/action.py

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional # <--  Optional 임포트

# ---  Pydantic 스키마 임포트 ---
from schemas.reasoning_output import ReasoningOutput # 입력 타입
from schemas.action_output import ActionOutput, MultilingualGuidelines # 반환 타입

# .env 파일 로드 및 OpenAI 클라이언트 초기화
load_dotenv()
api_key_from_env = os.environ.get("OPENAI_API_KEY")
if api_key_from_env:
    client = OpenAI(api_key=api_key_from_env)
else:
    print("🚨 [Action] 오류: OPENAI_API_KEY 환경 변수를 찾을 수 없습니다.")
    client = None

# ---  함수 시그니처에 Pydantic 타입 힌트 적용 ---
def generate_safety_guideline(analysis_result: ReasoningOutput) -> ActionOutput:
    """
    [실제 Action Layer 함수]
    VLM 분석 결과를 바탕으로 위험 등급에 따라 조치를 취하고,
    HIGH 위험 시 LLM을 호출하여 다국어 안전 지침을 생성합니다.
    """
    if not client:
        print("🚫 [Action] OpenAI 클라이언트가 초기화되지 않아 LLM 호출을 건너<0xEB><0><0xA4>니다.")
        # ---  Pydantic 모델로 반환 ---
        return ActionOutput(
            status="error_client_not_initialized",
            risk_level_processed=analysis_result.risk_level,
            hazard_code_processed=analysis_result.hazard_code,
            reason_detected=analysis_result.reason
        )

    # ---  Pydantic 객체 속성으로 접근 ---
    risk_level = analysis_result.risk_level
    reason = analysis_result.reason
    hazard_code = analysis_result.hazard_code

    print(f"📢 [Action] 위험 등급 '{risk_level}'(코드: {hazard_code})에 따른 조치 실행...")

    # ---  모든 반환값을 ActionOutput Pydantic 모델로 감싸기 ---
    if risk_level == "LOW":
        print("➡️ [Action] 위험 등급 LOW: 로그 기록만 수행합니다.")
        # 여기에 로그 기록 로직 추가 (예: 파일 저장, DB 저장 등)
        return ActionOutput(
            status="logged",
            risk_level_processed=risk_level,
            hazard_code_processed=hazard_code,
            reason_detected=reason
        )

    elif risk_level == "MED":
        print("⚠️ [Action] 위험 등급 MED: 확인 알림 표시.")
        print(f"   - 확인 필요: {reason} (위험 코드: {hazard_code})")
        # 여기에 확인 알림 UI 표시 또는 메시지 전송 로직 추가
        return ActionOutput(
            status="confirmation_requested",
            risk_level_processed=risk_level,
            hazard_code_processed=hazard_code,
            reason_detected=reason
        )

    elif risk_level == "HIGH":
        print("🚨 [Action] 위험 등급 HIGH: LLM 호출하여 다국어 안전 지침 생성...")

        # --- (수정 2) LLM 프롬프트에 hazard_code와 reason을 동적으로 삽입 ---
        prompt_for_llm = f"""
        다음은 스마트 공장에서 감지된 심각한 안전 위험 상황입니다:

        [확인된 위험 정보]
        * 위험 코드: {hazard_code}
        * 상황 설명: {reason}

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
            # LLM API 호출
            response = client.chat.completions.create(
                model="gpt-4o", 
                messages=[
                    {"role": "user", "content": prompt_for_llm}
                ],
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            guideline_str = response.choices[0].message.content
            guidelines_dict = json.loads(guideline_str)
            
            # ---  LLM 결과를 Pydantic 모델(MultilingualGuidelines)로 변환 ---
            output_guidelines = MultilingualGuidelines(**guidelines_dict)
            output_status = "multilingual_guideline_generated"

            print("--- [생성된 다국어 안전 지침] ---")
            print(f"🇰🇷 (한국어): {output_guidelines.guideline_ko}")
            print(f"🇺🇸 (English): {output_guidelines.guideline_en}")
            print(f"🇻🇳 (Tiếng Việt): {output_guidelines.guideline_vi}")
            print("---------------------------------")
            
            # ---  최종 결과를 ActionOutput 모델로 반환 ---
            return ActionOutput(
                status=output_status,
                risk_level_processed=risk_level,
                hazard_code_processed=hazard_code,
                reason_detected=reason,
                guidelines=output_guidelines
            )

        except Exception as e:
            print(f"🚨 [Action] LLM API 호출 또는 Pydantic 변환 오류: {e}")
            return ActionOutput(
                status=f"error_llm_api_call",
                risk_level_processed=risk_level,
                hazard_code_processed=hazard_code,
                reason_detected=reason
            )

    else:
        print(f"❓ [Action] 알 수 없는 위험 등급: {risk_level}")
        return ActionOutput(
            status="unknown_risk_level",
            risk_level_processed=risk_level,
            hazard_code_processed=hazard_code,
            reason_detected=reason
        )