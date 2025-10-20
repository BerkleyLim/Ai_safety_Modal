# src/monitoring/__init__.py

from ultralytics import YOLO
import torch

# 1. YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

# GPU 사용 가능 여부를 확인하고 모델을 해당 장치로 보냅니다.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"🤖 [Monitoring] YOLOv8 모델을 '{device}' 장치에서 실행합니다.")


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
        # result.boxes 객체에는 모든 탐지 정보가 들어있습니다.
        boxes = result.boxes
        
        for i in range(len(boxes)):
            # 클래스 이름 (예: 'person', 'car')
            class_id = int(boxes.cls[i])
            class_name = model.names[class_id]
            
            # 신뢰도 점수 (0~1 사이의 값)
            confidence = float(boxes.conf[i])
            
            # 바운딩 박스 좌표 [x_min, y_min, x_max, y_max]
            box = boxes.xyxy[i].cpu().numpy().tolist()
            
            detected_objects_list.append({
                "class": class_name,
                "confidence": confidence,
                "box": box
            })

    # 4. 이상 상황 판단 로직 (연구의 핵심 부분)
    # ---------------------------------------------------
    # 현재는 '사람(person)'이 1명이라도 탐지되면 '이상 상황'으로 간주합니다.
    # 향후 이 부분을 "안전모를 쓰지 않은 사람", "금지 구역에 들어온 사람" 등으로 고도화해야 합니다.
    detected_class_names = [obj['class'] for obj in detected_objects_list]
    print("detected_class_name",detected_class_names)
    if 'person' in detected_class_names:
        status = "anomaly_detected"
        print("✅ [Monitoring] 'person' 객체 탐지! 이상 상황으로 판단합니다.")
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