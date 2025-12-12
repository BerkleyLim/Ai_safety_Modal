# src/monitoring/__init__.py

from ultralytics import YOLO
import torch
from pathlib import Path
from typing import List
import sys
import os

# 프로젝트 루트 경로 설정 (import 문제 방지)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

# --- [중요] Pydantic 스키마 임포트 ---
try:
    from src.schemas.monitoring_output import MonitoringOutput, DetectedObject
except ImportError:
    # 실행 위치에 따라 경로가 다를 경우 대비
    from schemas.monitoring_output import MonitoringOutput, DetectedObject

MODELS_DIR = project_root / "models"

def _find_best_model():
    """models/ 폴더에서 가장 최근 학습된 best.pt 찾기"""
    if not MODELS_DIR.exists():
        return None

    best_models = list(MODELS_DIR.rglob("weights/best.pt"))
    if not best_models:
        return None

    # 가장 최근 수정된 모델 선택
    best_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(best_models[0])


# 1. YOLOv8 모델 로드 (학습된 모델 우선, 없으면 기본 모델)
custom_model_path = _find_best_model()
if custom_model_path:
    model = YOLO(custom_model_path)
    print(f"🤖 [Monitoring] 학습된 커스텀 모델 로드: {custom_model_path}")
else:
    model = YOLO('yolov8n.pt')
    print("🤖 [Monitoring] 기본 YOLOv8 모델 로드 (yolov8n.pt)")

# GPU 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu' and torch.backends.mps.is_available():
    device = 'mps' 

model.to(device)
print(f"🤖 [Monitoring] YOLOv8 모델을 '{device}' 장치에서 실행합니다.")

# 위험 클래스 정의 (커스텀 모델용) - 57개 클래스 중 UA/UC 클래스
# # AI Hub 공식 정의 기준
ANOMALY_CLASSES = [
    # Unsafe Action (UA) - 위험 행동 13개
    "forklift_blind_spot",       # UA-01: 지게차 시야 미확보
    "forklift_obstacle_nearby",  # UA-02: 지게차 적재 시 장애물
    "stacking_3_levels_flat",    # UA-03: 3단 이상 평치 적재
    "rack_improper_stacking",    # UA-04: 랙 적재상태 불량
    "unstable_cargo_loading",    # UA-05: 운반장비 불안정 적재
    "cargo_collapse",            # UA-06: 화물 붕괴
    "person_in_forklift_path",   # UA-10: 지게차 통로에 사람
    "forklift_safety_violation", # UA-12: 지게차 안전수칙 미준수
    "forklift_cargo_collapse",   # UA-13: 지게차 화물 붕괴
    "worker_in_forklift_zone",   # UA-14: 지게차 구역 내 작업자
    "pallet_truck_over_stacking",# UA-16: 핸드파레트카 과적재
    "flammable_in_welding_zone", # UA-17: 용접구역 가연물 침범
    "smoking_in_no_smoke_zone",  # UA-20: 비흡연구역 흡연
    # Unsafe Condition (UC) - 위험 상태 15개
    "worker_in_truck_loading",   # UC-02: 입고 시 트럭 내 작업자
    "worker_in_truck_unloading", # UC-06: 출고 시 트럭 내 작업자
    "forklift_path_unmarked",    # UC-08: 지게차 통로 미표시
    "dock_door_obstacle",        # UC-09: 도크 출입문 장애물
    "person_behind_docking",     # UC-10: 도크 접차 시 후방 사람
    "pallet_disorganized",       # UC-13: 빈 파렛트 미정돈
    "worker_leaning_on_rack",    # UC-14: 랙에 기대는 작업자
    "pallet_damaged",            # UC-15: 파렛트 파손
    "worker_in_elevator",        # UC-16: 화물승강기 탑승
    "no_surge_protector",        # UC-17: 과부하차단 없는 멀티탭
    "no_fire_extinguisher",      # UC-18: 소화기 미비치
    "restricted_door_open",      # UC-19: 출입제한구역 문 열림
    "cargo_in_fire_escape",      # UC-20: 화재대피로 적재물
    "truck_dock_separated",      # UC-21: 도크-트럭 분리
    "forklift_outside_path",     # UC-22: 지게차 영역 이탈
    # --- SO (위험 관련 객체 추가) ---
    "floor_contaminant",  # 바닥 이물질 (방금 로그에 뜬 것)
    "flammable_material", # 가연물
    "smoking"             # 흡연
]

def detect_objects(image_path: str) -> MonitoringOutput:
    """
    [실제 Monitoring Layer 함수]
    YOLOv8 모델을 사용하여 이미지에서 객체를 탐지하고,
    미리 정의된 규칙에 따라 '이상 상황' 여부를 판단합니다.

    Args:
        image_path (str): 분석할 이미지 파일의 경로

    Returns:
        dict: 탐지 결과와 상태 정보를 담은 딕셔너리
    """
    print(f"👀 [Monitoring] YOLOv8 모델로 '{image_path}'의 객체 탐지를 시작합니다...")

   # 2. 모델을 사용하여 이미지 추론 실행
    results = model(image_path)
    results = model(image_path, conf=0.15)
    # --- Pydantic 객체 리스트 생성 ---
    pydantic_detected_objects: List[DetectedObject] = []
    
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            class_name = model.names[class_id]
            confidence = float(boxes.conf[i])
            box = boxes.xyxy[i].cpu().numpy().tolist()
            
            # DetectedObject 객체 생성 및 추가
            pydantic_detected_objects.append(
                DetectedObject(
                    class_name=class_name, # alias='class'
                    confidence=confidence,
                    box=box
                )
            )

    # 4. 이상 상황 판단 로직
    detected_class_names = [obj.class_name for obj in pydantic_detected_objects]
    print(f"detected_class_names: {detected_class_names}")
    
    # 탐지된 클래스 중 하나라도 위험 목록에 있으면 True
    is_anomaly = any(cls in ANOMALY_CLASSES for cls in detected_class_names)
    
    if is_anomaly:
        status = "anomaly_detected"
        # 감지된 위험 요소 출력
        dangers = [cls for cls in detected_class_names if cls in ANOMALY_CLASSES]
        print(f"✅ [Monitoring] 위험 감지됨! ({', '.join(dangers)}) -> Reasoning Layer 호출")
    else:
        status = "normal"
        print("➡️ [Monitoring] 특이사항 없음. 파이프라인을 종료합니다.")

    # --- 최종 결과를 MonitoringOutput 객체로 반환 ---
    return MonitoringOutput(
        status=status,
        image_path=str(image_path),
        detected_objects=pydantic_detected_objects
    )