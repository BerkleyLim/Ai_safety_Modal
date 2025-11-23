# 모니터링 모듈

YOLOv8 모델을 사용하여 이미지에서 객체를 탐지하고 이상 상황을 판단합니다.

## 사용법

```python
from monitoring import detect_objects

result = detect_objects("image.jpg")
print(result["status"])  # "anomaly_detected" 또는 "normal"
```

## 모델 로드 우선순위

1. `models/` 폴더에서 학습된 `best.pt` 자동 탐색
2. 없으면 기본 `yolov8n.pt` 사용

## 이상 판단 로직

- **커스텀 모델**: 위험 클래스 탐지 시 이상 (`no_helmet`, `pathway_obstacle` 등)
- **기본 모델**: `person` 탐지 시 이상