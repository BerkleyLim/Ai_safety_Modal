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
        high_conf_events = [obj.class_name for obj in detection_result.detected_objects if obj.confidence > 0.2]
        if high_conf_events:
            detected_events_str = ", ".join(high_conf_events)
    
    # [YOLO_DETECTED_EVENTS] 부분을 실제 탐지된 이벤트 문자열로 교체
    prompt_template = """
        당신은 물류창고의 AI 안전 관리 시스템(Safety Officer)입니다.
        당신의 임무는 CCTV 이미지를 분석하여, 아래 [안전 점검 목록] 중에서 현재 상황에 해당하는 위험을 식별하고 그 심각성을 평가하는 것입니다.

        [상황]
        첨부된 이미지는 물류창고 CCTV 화면입니다.
        1차 분석 시스템(YOLO)이 이 이미지에서 {{ {dynamic_detected_events} }} 을(를) 탐지했습니다.
        이 정보는 ‘강력한 단서(hint)’이지만, YOLO는 오류(FP/FN/클래스 혼동)가 있을 수 있으므로 
        반드시 이미지와 교차검증 후 최종 판단하십시오.

        [YOLO class_name 의미 및 위험 코드 매핑]
        SO/WO 객체 = 맥락 정보 (단독으로는 위험이 아님)
        예) storage_rack, forklift_path, worker_uniform, stacked_cargo_group, cargo_individual 등

        2) 위험 후보 객체 (UA/UC/SO-21/SO-22)
        다음 class_name이 탐지되면, 해당 위험 코드의 “후보”입니다.
        단, 이미지에서 그에 해당하는 상황이 명확히 보이지 않으면 **사용하지 않습니다.**

        - forklift_blind_spot           → UA-01 (지게차 운전자 시야 미확보)
        - forklift_obstacle_nearby      → UA-02 (지게차 적재 시 주변 장애물)
        - stacking_3_levels_flat        → UA-03 (3단 이상 평치 적재)
        - rack_improper_stacking        → UA-04 (랙 적재상태 불량)
        - unstable_cargo_loading        → UA-05 (운반수레 등 적재물 불안정)
        - cargo_collapse                → UA-06 (운반 중 화물 붕괴)
        - person_in_forklift_path       → UA-10 (지게차 통로에 사람)
        - forklift_safety_violation     → UA-12 (지게차 안전수칙 위반)
        - forklift_cargo_collapse       → UA-13 (지게차 화물 적재 불량/붕괴)
        - worker_in_forklift_zone       → UA-14 (지게차 전용 구역 내 작업자)
        - pallet_truck_over_stacking    → UA-16 (핸드카 2단 이상 적재)
        - flammable_in_welding_zone     → UA-17 (용접구역 가연물)
        - smoking_in_no_smoke_zone      → UA-20 (금연 구역 흡연)

        - worker_in_truck_loading       → UC-02 (트럭 내 작업자 – 입고)
        - worker_in_truck_unloading     → UC-06 (트럭 내 작업자 – 출고)
        - forklift_path_unmarked        → UC-08 (지게차 이동 통로 미표시)
        - dock_door_obstacle            → UC-09 (도크문 앞 장애물)
        - person_behind_docking         → UC-10 (도크 후진 차량 뒤 사람)
        - pallet_disorganized           → UC-13 (빈 파렛트 미정돈)
        - worker_leaning_on_rack        → UC-14 (랙에 기대는 작업자)
        - pallet_damaged                → UC-15 (파렛트 파손)
        - worker_in_elevator            → UC-16 (화물승강기 탑승자)
        - no_surge_protector            → UC-17 (차단기 없는 멀티탭)
        - no_fire_extinguisher          → UC-18 (소화기 미비치)
        - restricted_door_open          → UC-19 (출입제한구역 문 개방)
        - cargo_in_fire_escape          → UC-20 (대피로 적재물)
        - truck_dock_separated          → UC-21 (트럭-도크 분리된 상태 작업)
        - forklift_outside_path         → UC-22 (지게차 안전선 밖 주행)

        - floor_contaminant             → SO-21 (바닥 이물질/기름)
        - flammable_material            → SO-22 (가연성/인화성 물질 방치)

        [YOLO 교차검증 규칙]
        - 위 UA/UC/SO-21/SO-22 class_name은 “후보”일 뿐이며, 이미지에 그 상황이 **명확하게** 보이지 않으면 해당 코드를 사용하지 마십시오.
        - YOLO가 탐지하지 않아도, 이미지에서 해당 위험이 명확하면 위험 코드를 선택합니다.
        - SO/WO는 맥락 정보이며 단독으로 위험이 되지 않습니다.
        - YOLO와 이미지 모두 위험 단서가 거의 없을 때만 SAFE입니다.

        [안전 점검 목록 (위험 Class ID)]
        아래 각 항목은 이름만이 아니라, 해당 위험이 인정되기 위한 핵심 조건(트리거)을 포함합니다.
        ---
        [안전 점검 목록 (위험 코드 및 최소 트리거)]

        ※ 아래 코드들은 “이 조건이 눈에 보일 때만” 사용합니다.
        “그럴 수도 있다/불안정해 보인다” 수준은 코드 선택 기준에 포함되지 않습니다.

        UA-01: 지게차 운전자 시야 미확보
        - 트리거: 지게차 전방이 화물·구조물 등으로 가려져 운전자의 전방 시야가 거의 보이지 않을 때

        UA-02: 지게차 적재 시 주변 장애물
        - 트리거: 시야는 확보되지만, 지게차 진행 방향 바로 주변(충돌 범위)에 팔레트·화물 등 장애물이 근접해 있을 때

        UA-03: 3단 이상 화물 평치 적재
        - 트리거: 팔레트·타이어 등 동일 물체가 **명확히 3단 이상** 수평으로 쌓인 것이 보일 때

        UA-04: 랙 화물 적재상태 불량
        - 트리거: 랙에서 화물이 심하게 삐져나오거나 기울어진 상태가 선명하게 보일 때

        UA-05: 운반수레 등 적재물 불안정
        - 트리거: 핸드카/팔레트 위 화물이 크게 기울어지거나 떨어질 듯 한 상태가 명확할 때

        UA-06: 운반 중 화물 붕괴
        - 트리거: 화물이 실제로 무너졌거나, 일부가 떨어진 모습이 보일 때

        UA-10: 지게차 이동 통로에 사람
        - 트리거: 지게차가 이동 중이며, 그 바로 앞이나 통로 위에 사람이 서 있거나 걷고 있을 때

        UA-12: 지게차 포크에 사람 탑승
        - 트리거: 지게차 포크 위에 사람이 올라타 있거나 매달린 모습이 보일 때

        UA-13: 지게차 화물 적재 불량/붕괴
        - 트리거: 화물 자체의 기울어짐과 붕괴에 초점으로 지게차 포크 위 화물이 명확히 기울어져 있거나, 일부가 떨어진 모습이 보일 때

        UA-14: 지게차 전용 구역 내 작업자
        - 트리거: 바닥에 표시된 지게차 전용 구역 안에 사람이 서 있거나 작업 중일 때

        UA-16: 핸드카 2단 이상 적재
        - 트리거: 핸드카/손수레 위에 상자/화물이 **확실히 2단 이상** 수직으로 쌓여 있을 때

        UA-17: 용접 등 화기 작업 구역 내 가연물
        - 트리거: 용접 토치·불꽃 근처에 종이박스·플라스틱 등 가연물이 가까이 놓인 것이 보일 때

        UA-20: 비 흡연 구역 내 흡연
        - 트리거: 사람이 담배/전자담배를 손에 들고 있거나 연기가 보이며, 금연 구역 표식이 함께 보일 때

        UC-02/UC-06: 트럭 화물칸 내 작업자
        - 트리거: 트럭 적재함 안에 사람이 서 있거나 지게차와 함께 작업하는 모습이 보일 때

        UC-08: 지게차 이동통로 미표시
        - 트리거: 시야 및 장애물 문제가 아닌, 지게차가 주행하는 바닥에 차선/통로 표시가 거의 없는 것이 넓은 영역에서 보일 때

        UC-09: 도크 출입문 앞 장애물
        - 트리거: 도크문 바로 앞(출입 영역)에 팔레트/화물이 막고 있는 것이 명확할 때

        UC-10: 도크 후진 차량 후방에 사람
        - 트리거: 도크로 후진 중인 트럭 바로 뒤에 사람이 서거나 이동 중인 모습이 보일 때

        UC-13: 빈 파렛트 미정돈
        - 트리거: 빈 파렛트가 바닥에 무질서하게 흩어져 있거나 쌓여 있는 모습이 보일 때

        UC-14: 랙에 기대는 작업자
        - 트리거: 사람이 몸을 랙에 기대거나 기대어 서 있는 모습이 보일 때

        UC-15: 파렛트 파손/부식
        - 트리거: 파렛트 판자가 부러져 있거나 형태가 명확히 찌그러진 것이 보일 때

        UC-16: 화물 승강기에 작업자 탑승
        - 트리거: 화물승강기 안에 작업자가 서 있거나 탑승 중인 것이 보일 때

        UC-17: 과부하 차단기 없는 멀티탭 사용
        - 트리거: 차단 스위치가 없는 단순 멀티탭에 다수의 플러그가 꽂힌 모습이 보일 때

        UC-18: 소화기 미비치
        - 트리거: 소화기가 있어야 할 위치(표식 등)에 소화기가 전혀 보이지 않을 때

        UC-19: 출입제한 구역 출입문 열림
        - 트리거: 출입제한 표식이 있는 문이 열린 상태로 보일 때

        UC-20: 화재 대피로 내 적재물
        - 트리거: 바닥/벽 표시로 나타난 대피로 위에 팔레트·화물이 놓여 있는 것이 명확할 때

        UC-21: 화물트럭-도크 분리 (작업 중)
        - 트리거: 트럭이 도크에 완전히 밀착되지 않은 상태에서 작업이 진행되는 것이 보일 때

        UC-22: 지게차 안전선 이탈 주행
        - 트리거: 지게차가 바닥에 표시된 안전선 밖에서 주행하는 모습이 분명할 때

        SO-21: 바닥 이물질/기름
        - 트리거: 바닥에 물/기름/얼룩 등 미끄러운 이물질이 **뚜렷하게** 보일 때

        SO-22: 가연성/인화성 물질 방치
        - 트리거: 드럼통, 액체 용기, 종이박스 등 가연성 물질이 바닥에 방치된 모습이 분명할 때
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
        중대성(S) x 가능성(L) = 리스크 점수
        - HIGH (III등급): 6 ~ 9점
        - MED  (II등급): 3 ~ 4점예: 
        - LOW  (I등급): 1 ~ 2점
        - SAFE: 위 위험 목록에 해당하는 사항이 전혀 발견되지 않은 안전한 상태
        
        [중요 판단 규칙]
        다음 규칙을 반드시 지키십시오.

        1. SAFE 판단 원칙
        - SAFE는 다음 두 조건을 모두 만족할 때만 사용합니다.
        (1) 이미지 상에서 [안전 점검 목록]의 어떤 항목과 트리거에 거의 해당하지 않는다고 판단되는 경우
        (2) YOLO 탐지 결과(`{{ {dynamic_detected_events} }}`)에도 위험과 직접적으로 연결되는 이벤트가 거의 없는 경우

        2. 상상 금지
        - CCTV 프레임 밖 상황, 가려진 영역, 과거/미래 상황을 추측하지 마십시오.
        - 실제 이미지에 명확히 보이는 정보만 근거로 사용합니다.

        3. 코드 선택 원칙
        - 먼저 [안전 점검 목록] 중에서, 현재 화면 상황과 가장 잘 일치하는 항목을 찾습니다.
        - 여러 항목이 동시에 해당될 수 있어 보이는 경우, 사고로 직접 이어질 가능성이 가장 큰 한 가지 코드만 선택합니다.
        - 어떤 코드와도 거의 맞지 않는다고 판단될 때만 SAFE로 판정합니다.

        4. YOLO 결과 활용 원칙
        - `{{ {dynamic_detected_events} }}`는 중요한 보조 정보입니다.
        - YOLO결과의 위험 객체와 그 외 정보를 맥락 보조 정보로 활용합니다. (단, 이미지와 모순 시 무효)
        - YOLO와 이미지 둘 다에서 위험 단서가 거의 보이지 않을 때만 SAFE로 판정합니다.

        [분석 지시]
        1. 상황 묘사: 먼저 이미지 내의 작업자, 장비, 사물의 배치와 상태를 객관적으로 묘사하십시오.
        2. 규칙 대조: YOLO결과와 묘사한 상황정보를 활용하여 [안전 점검 목록]의 항목 및 트리거 참고하여 하나와 일치하는지 확인하십시오. 위험 상황 발생 시 가장 최초의 원인이 되는 항목으로 확인하세요.
        3. 예외 처리: 육안으로 명확한 위반 사항이나 위험 요소가 없다면, 위험 등급을 **'SAFE'**로 판정하고 분석을 종료하십시오.
        4. 위험 평가: 위반 사항이 확실히 존재할 때만, 가장 핵심적인 위험 1가지를 식별하고 그 위험의 **중대성(S1~S3)**과 **가능성(L1~L3)**을 판단하여 등급(LOW/MED/HIGH)을 결정하십시오. 
        5. 근거 설명: 판단 근거를 설명할 때, **"어떤 상황이라 S몇 등급이고, 어떤 요소 때문에 L몇 등급으로 보아 최종 XX 등급으로 판단했다"**는 논리를 한국어로 명확히 서술하십시오.

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
        # 1. hazard_code 추출 (없으면 "NONE"으로 처리)
        # VLM이 실수로 키 이름을 다르게 줄 경우를 대비해 여러 키 시도
        hazard_code = analysis_result_dict.get("hazard_code")
        if not hazard_code:
            hazard_code = analysis_result_dict.get("Hazard_Code") # 대문자 키 등 대비
        
        if not hazard_code:
            hazard_code = "NONE"
        
        # 2. risk_level이 SAFE면 hazard_code도 NONE으로 강제 통일
        if analysis_result_dict.get("risk_level") == "SAFE":
            hazard_code = "NONE"

        # 3. 딕셔너리에 확정된 값 다시 저장
        analysis_result_dict["hazard_code"] = hazard_code

        # ---------------------------------------------------------
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