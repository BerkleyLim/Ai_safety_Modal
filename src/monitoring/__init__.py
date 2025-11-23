# src/monitoring/__init__.py

from ultralytics import YOLO
import torch
from pathlib import Path

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


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

# GPU 사용 가능 여부를 확인하고 모델을 해당 장치로 보냅니다.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"🤖 [Monitoring] YOLOv8 모델을 '{device}' 장치에서 실행합니다.")

# 위험 클래스 정의 (커스텀 모델용) - 57개 클래스 중 UA/UC 클래스
# AI Hub 공식 정의 기준
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
]


def detect_objects(image_path):
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

    detected_objects_list = []

    # 3. 탐지 결과에서 필요한 정보 추출
    for result in results:
        boxes = result.boxes

        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            class_name = model.names[class_id]
            confidence = float(boxes.conf[i])
            box = boxes.xyxy[i].cpu().numpy().tolist()

            detected_objects_list.append({
                "class": class_name,
                "confidence": confidence,
                "box": box
            })

    # 4. 이상 상황 판단 로직
    # ---------------------------------------------------
    detected_class_names = [obj['class'] for obj in detected_objects_list]
    print("detected_class_names:", detected_class_names)

    # 커스텀 모델: 위험 클래스가 탐지되면 이상 상황
    # 기본 모델: person이 탐지되면 이상 상황
    anomaly_detected = False

    if custom_model_path:
        # 커스텀 모델 - 위험 클래스 체크
        for class_name in detected_class_names:
            if class_name in ANOMALY_CLASSES:
                anomaly_detected = True
                print(f"⚠️ [Monitoring] 위험 상황 탐지: {class_name}")
                break
    else:
        # 기본 모델 - person 체크 (기존 로직)
        if 'person' in detected_class_names:
            anomaly_detected = True

    if anomaly_detected:
        status = "anomaly_detected"
        print("✅ [Monitoring] 이상 상황으로 판단합니다.")
    else:
        status = "normal"
        print("➡️ [Monitoring] 특이사항 없음. 파이프라인을 종료합니다.")
    # ---------------------------------------------------

    # 최종 결과를 딕셔너리 형태로 정리하여 반환
    return {
        "status": status,
        "image_path": image_path,
        "detected_objects": detected_objects_list
    }