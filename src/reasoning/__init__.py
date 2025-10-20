# src/reasoning/__init__.py

import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 환경 변수에서 API 키를 가져와 클라이언트를 초기화합니다.
# 이 코드는 모듈이 임포트될 때 한 번 실행됩니다.
api_key_from_env = os.environ.get("OPENAI_API_KEY")
if api_key_from_env:
    client = OpenAI(api_key=api_key_from_env)
else:
    print("🚨 [Reasoning] 오류: OPENAI_API_KEY 환경 변수를 찾을 수 없습니다.")
    client = None # 클라이언트 초기화 실패

def encode_image_to_base64(image_path):
    """이미지를 Base64 문자열로 인코딩하는 헬퍼 함수"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"🚨 [Reasoning] 오류: 이미지 파일을 찾을 수 없습니다 - {image_path}")
        return None

def analyze_risk_with_vlm(detection_result):
    """
    [실제 Reasoning Layer 함수]
    GPT-4o VLM API를 호출하여 이미지의 위험 수준과 이유를 분석합니다.
    """
    if not client:
        print("🚫 [Reasoning] OpenAI 클라이언트가 초기화되지 않아 API 호출을 건너<0xEB><0><0xA4>니다.")
        return None # 클라이언트 없으면 실행 불가

    print("🧠 [Reasoning] GPT-4o VLM으로 위험 상황을 심층 분석 중...")

    image_path = detection_result.get("image_path")
    if not image_path:
        print("🚨 [Reasoning] 오류: detection_result에 'image_path'가 없습니다.")
        return None

    # VLM에 전송할 이미지를 Base64로 인코딩합니다.
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None # 이미지 인코딩 실패

    # VLM에게 내릴 지시사항 (프롬프트)을 작성합니다.
    prompt_text = """
    당신은 제조 공장의 AI 안전 관리자입니다.
    첨부된 이미지는 공장 CCTV 화면입니다. 이미지를 면밀히 분석하여 잠재적인 안전 위험을 평가해주세요.

    - 위험 수준(risk_level)은 "LOW"(안전), "MED"(주의), "HIGH"(높은 위험) 중 하나로 판단해야 합니다.
    - 위험을 판단한 구체적인 이유(reason)를 한국어로 한 문장으로 명확하게 설명해주세요.

    결과는 반드시 아래와 같은 JSON 형식으로만 응답해야 합니다:
    {"risk_level": "...", "reason": "..."}
    """

    try:
        # OpenAI API를 호출합니다.
        response = client.chat.completions.create(
            model="gpt-4o", # 또는 "gpt-4o-mini" 등 사용 가능한 모델
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            # 이미지 타입에 맞게 jpeg 또는 png 지정
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=300,
            response_format={"type": "json_object"} # 응답을 JSON 형식으로 강제
        )

        # API 응답에서 분석 결과를 추출합니다.
        analysis_result_str = response.choices[0].message.content
        analysis_result = json.loads(analysis_result_str)

        print(f"✅ [Reasoning] VLM 분석 완료: {analysis_result}")
        return analysis_result

    except Exception as e:
        print(f"🚨 [Reasoning] VLM API 호출 중 오류 발생: {e}")
        return None