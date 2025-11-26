# src/reasoning/__init__.py

import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional # <--  Optional 임포트

# ---  Pydantic 스키마 임포트 ---
from schemas.monitoring_output import MonitoringOutput # 입력 타입
from schemas.reasoning_output import ReasoningOutput   # 반환 타입

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 환경 변수에서 API 키를 가져와 클라이언트를 초기화합니다.
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
    except Exception as e:
        print(f"🚨 [Reasoning] Base64 인코딩 오류: {e}")
        return None

# ---  함수 시그니처에 Pydantic 타입 힌트 적용 ---
def analyze_risk_with_vlm(detection_result: MonitoringOutput) -> Optional[ReasoningOutput]:
    """
    [실제 Reasoning Layer 함수]
    GPT-4o VLM API를 호출하여 이미지의 위험 수준과 이유를 분석합니다.
    """
    if not client:
        print("🚫 [Reasoning] OpenAI 클라이언트가 초기화되지 않아 API 호출을 건너<0xEB><0><0xA4>니다.")
        return None # 클라이언트 없으면 실행 불가

    print("🧠 [Reasoning] GPT-4o VLM으로 위험 상황을 심층 분석 중...")

    # ---  Pydantic 객체 속성으로 접근 ---
    image_path = detection_result.image_path 
    if not image_path:
        print("🚨 [Reasoning] 오류: detection_result에 'image_path'가 없습니다.")
        return None

    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None # 이미지 인코딩 실패

    # --- (수정 2) Monitoring Layer의 탐지 결과를 프롬프트에 동적으로 삽입 ---
    detected_events_str = "탐지된 이벤트 없음"
    if detection_result.detected_objects:
        # (예시) 신뢰도 0.5 이상인 것들만 클래스 이름 목록으로 만듦
        high_conf_events = [obj.class_name for obj in detection_result.detected_objects if obj.confidence > 0.5]
        if high_conf_events:
            detected_events_str = ", ".join(high_conf_events)
    
    # [YOLO_DETECTED_EVENTS] 부분을 실제 탐지된 이벤트 문자열로 교체
    prompt_template = """
        당신은 물류창고의 AI 안전 관리 시스템(Safety Officer)입니다.
        당신의 임무는 CCTV 이미지를 분석하여, 아래 [안전 점검 목록] 중에서 현재 상황에 해당하는 위험을 식별하고 그 심각성을 평가하는 것입니다.

        [상황]
        첨부된 이미지는 물류창고 CCTV 화면입니다.
        1차 분석 시스템(YOLO)이 이 이미지에서 {{ {dynamic_detected_events} }} 을(를) 탐지했습니다.
        이 정보를 참고하여, 이미지의 전체적인 맥락을 심층 분석하십시오.

        [안전 점검 목록 (위험 Class ID)]
        ---
        * UA-01: 지게차 운전자 시야 미확보
        * UA-02: 지게차 적재 시 주변 장애물
        * UA-03: 3단 이상 화물 평치 적재
        * UA-04: 랙 화물 적재상태 불량
        * UA-05: 운반수레 등 적재물 불안정
        * UA-06: 운반 중 화물 붕괴
        * UA-10: 지게차 이동 통로에 사람
        * UA-12: 지게차 포크에 사람 탑승
        * UA-13: 지게차 화물 적재 불량/붕괴
        * UA-14: 지게차 전용 구역 내 작업자
        * UA-16: 핸드카 2단 이상 적재
        * UA-17: 용접 등 화기 작업 구역 내 가연물
        * UA-20: 비 흡연 구역 내 흡연
        * UC-02: 트럭 화물칸 내 작업자 (지게차 입고)
        * UC-06: 트럭 화물칸 내 작업자 (지게차 출고)
        * UC-08: 지게차 이동통로 미표시
        * UC-09: 도크 출입문 앞 장애물
        * UC-10: 도크 후진 차량 후방에 사람
        * UC-13: 빈 파렛트 미정돈
        * UC-14: 랙 안전선 내 랙에 기댄 작업자
        * UC-15: 파렛트 파손/부식
        * UC-16: 화물 승강기에 작업자 탑승
        * UC-17: 과부하 차단기 없는 멀티탭 사용
        * UC-18: 소화기 미비치
        * UC-19: 출입제한 구역 출입문 열림
        * UC-20: 화재 대피로 내 적재물
        * UC-21: 화물트럭-도크 분리 (작업 중)
        * UC-22: 지게차 안전선 이탈 주행
        ---

        [리스크 평가 개념]
        본 평가는 **KOSHA GUIDE X-66-2013**의 “중대성 × 가능성” 개념을 적용합니다.

        1. **중대성(Severity, S)**: 사고 발생 시 피해 크기
        - **S3 (대)**: 사망, 영구장애, 대형 화재/폭발.
        - **S2 (중)**: 골절 등 휴업이 필요한 부상.
        - **S1 (소)**: 찰과상 등 경미한 부상.

        2. **가능성(Likelihood, L)**: 사고 발생 빈도
        - **L3 (높음)**: 현재 상태 지속 시 사고 발생 가능성 매우 높음.
        - **L2 (보통)**: 부주의 등 조건이 겹치면 사고 가능성 있음.
        - **L1 (낮음)**: 잠재적 위험은 있으나 당장 사고 확률은 낮음.

        [위험 등급 평가 기준 (Risk Level)]
        위의 S와 L을 종합하여 최종 등급을 결정하십시오.
        * **HIGH (즉각 조치)**: (S3) 이거나 (S2 + L3) 인 경우. (예: 추락/협착/충돌 직전, 대형 화재 위험)
        * **MED (경고/시정)**: (S2) 이거나 (S1 + L3) 인 경우. (예: 안전수칙 위반, 불안전한 적재)
        * **LOW (점검/개선)**: (S1) 이거나 (L1) 인 경우. (예: 단순 정리정돈 미흡)

        [분석 지시]
        1. 이미지에서 [안전 점검 목록]에 위배되는 **가장 핵심적인 위험 1가지**를 식별하십시오.
        2. 해당 위험의 **중대성(S1~S3)**과 **가능성(L1~L3)**을 판단하십시오.
        3. 이를 종합하여 위험 등급을 **'LOW', 'MED', 'HIGH'** 중 하나로 분류하십시오.
        4. 판단 근거를 설명할 때, **"어떤 상황이라 S몇 등급이고, 어떤 요소 때문에 L몇 등급으로 보아 최종 XX 등급으로 판단했다"**는 논리를 한국어로 명확히 서술하십시오.

        [출력 형식]
        결과는 반드시 다음 JSON 형식으로만 응답하십시오:
        {{
        "risk_level": "분류한 위험 등급 (LOW, MED, 또는 HIGH)",
        "hazard_code": "식별된 가장 핵심적인 위험의 Class ID (예: UA-17)",
        "reason": "위험 등급 및 hazard_code 판단 근거 (중대성, 가능성 평가 포함된 한국어 설명)"
        }}
        """
    # f-string을 사용해 프롬프트의 {dynamic_detected_events} 부분을 채웁니다.
    prompt_text = prompt_template.format(dynamic_detected_events=detected_events_str)
    
    # -----------------------------------------------------------------

    try:
        # OpenAI API를 호출합니다.
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"} # 확장자에 맞게 jpeg/png 수정
                        }
                    ]
                }
            ],
            max_tokens=300,
            response_format={"type": "json_object"} 
        )

        analysis_result_str = response.choices[0].message.content
        analysis_result_dict = json.loads(analysis_result_str)

        # ---  Pydantic 모델로 변환하여 반환 ---
        # VLM의 JSON 응답과 image_path를 합쳐 ReasoningOutput 객체 생성
        output = ReasoningOutput(
            image_path=image_path, 
            **analysis_result_dict # VLM이 반환한 risk_level, hazard_code, reason이 여기에 포함됨
        )
        
        print(f"✅ [Reasoning] VLM 분석 완료: {output.model_dump(mode='json')}")
        return output

    except Exception as e:
        print(f"🚨 [Reasoning] VLM API 호출 또는 Pydantic 변환 오류: {e}")
        return None