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

# 위험 클래스 정의 (커스텀 모델용)
ANOMALY_CLASSES = [
    "no_helmet", "no_safety_shoes", "no_safety_vest", "danger_zone_entry",
    "phone_while_driving", "speeding", "other_unsafe_action",
    "pathway_obstacle", "improper_stacking", "poor_lighting", "other_unsafe_condition"
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