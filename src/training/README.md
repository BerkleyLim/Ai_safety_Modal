# YOLO 모델 학습 모듈

물류창고 안전 데이터셋으로 YOLOv8 객체 탐지 모델을 학습합니다.

## 요구사항

```bash
pip install ultralytics
```

## 빠른 시작

### 1. 전처리 완료 확인

학습 전에 데이터 전처리가 완료되어 있어야 합니다:

```bash
# 전처리 실행
cd src
python run.py --mode preprocess --folders 01
```

전처리 후 다음 구조가 있어야 합니다:

```
data/01_도크설비/logistics_yolo/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

### 2. 학습 실행

```bash
# 카테고리 01 학습 (기본 설정)
python -m src.training.train_yolo --category 01

# 커스텀 설정
python -m src.training.train_yolo \
    --category 01 \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --model yolov8n.pt

#커스텀 설정 (데이터경로 다를 경우)
python src/training/train_yolo.py \
  --category 05 \
  --data-root "/Volumes/Elements/data" \
  --epochs 100 \
  --batch 16 \
  --device mps \
  --model yolov8n.pt

# GPU 지정
python -m src.training.train_yolo --category 01 --device 0

# 여러 GPU 사용
python -m src.training.train_yolo --category 01 --device 0,1
```

### 3. run.py에서 실행

```bash
cd src
python run.py --mode train --category 01 --epochs 50
```

---

## 명령줄 옵션

| 옵션          | 기본값     | 설명                             |
| ------------- | ---------- | -------------------------------- |
| `--category`  | 01         | 학습할 카테고리 (01~11 또는 all) |
| `--epochs`    | 100        | 학습 에포크 수                   |
| `--batch`     | 16         | 배치 크기                        |
| `--imgsz`     | 640        | 이미지 크기                      |
| `--patience`  | 50         | Early stopping patience          |
| `--workers`   | 8          | 데이터 로더 워커 수              |
| `--model`     | yolov8n.pt | 사전학습 모델                    |
| `--device`    | auto       | 학습 장치                        |
| `--resume`    | -          | 이전 학습 재개                   |
| `--data-root` | -          | 데이터 경로(optional)            |

---

## 사전학습 모델 선택

| 모델       | 파라미터 | mAP  | 속도 | 용도        |
| ---------- | -------- | ---- | ---- | ----------- |
| yolov8n.pt | 3.2M     | 낮음 | 빠름 | 테스트/엣지 |
| yolov8s.pt | 11.2M    | 중하 | 빠름 | 경량 배포   |
| yolov8m.pt | 25.9M    | 중간 | 보통 | 균형        |
| yolov8l.pt | 43.7M    | 중상 | 느림 | 정확도 중시 |
| yolov8x.pt | 68.2M    | 높음 | 느림 | 최고 정확도 |

---

## Python에서 사용

```python
from src.training import train_yolo, YOLOTrainer

# 간편 함수
result = train_yolo(
    category='01',
    epochs=100,
    batch=16,
    imgsz=640
)

# 클래스 사용
trainer = YOLOTrainer(model_name='yolov8s.pt')
result = trainer.train(category='01', epochs=100)

# 검증
metrics = trainer.validate(weights_path='models/best.pt')
print(f"mAP50: {metrics['mAP50']}")

# 모델 내보내기
trainer.export(weights_path='models/best.pt', format='onnx')
```

---

## 카테고리 목록

| 번호 | 이름           | 설명             |
| ---- | -------------- | ---------------- |
| 01   | 도크설비       | 도크 관련 안전   |
| 02   | 보관           | 보관 구역 안전   |
| 03   | 부가가치서비스 | VAS 작업 안전    |
| 04   | 설비및장비     | 장비 관련 안전   |
| 05   | 운반           | 운반 작업 안전   |
| 06   | 입고           | 입고 작업 안전   |
| 07   | 지게차         | 지게차 관련 안전 |
| 08   | 출고           | 출고 작업 안전   |
| 09   | 파렛트렉       | 파렛트/렉 안전   |
| 10   | 피킹분배       | 피킹 작업 안전   |
| 11   | 화재           | 화재 관련 안전   |
| all  | 전체           | 통합 학습        |

---

## 출력 구조

학습 완료 후 `models/` 폴더에 결과가 저장됩니다:

```
models/
└── safety_01_도크설비_20241122_120000/
    ├── weights/
    │   ├── best.pt      # 최고 성능 모델
    │   └── last.pt      # 최종 모델
    ├── args.yaml        # 학습 설정
    ├── results.csv      # 에포크별 결과
    ├── results.png      # 학습 그래프
    ├── confusion_matrix.png
    ├── F1_curve.png
    ├── PR_curve.png
    └── ...
```

---

## 학습 팁

### 1. 배치 크기

- GPU 메모리에 따라 조절
- RTX 3080 (10GB): batch=16
- RTX 4090 (24GB): batch=32

### 2. 이미지 크기

- 640: 일반적인 선택
- 1280: 작은 객체 탐지 시

### 3. 에포크

- 100~300: 일반적
- Early stopping이 있어서 과적합 방지

### 4. 데이터 증강

- YOLO는 자동으로 데이터 증강 적용
- mosaic, mixup, hsv 변환 등

---

## 문제 해결

### CUDA out of memory

```bash
# 배치 크기 줄이기
python -m src.training.train_yolo --category 01 --batch 8
```

### data.yaml을 찾을 수 없음

```bash
# 전처리 먼저 실행
python run.py --mode preprocess --folders 01
```

### 학습이 너무 느림

```bash
# 워커 수 늘리기
python -m src.training.train_yolo --category 01 --workers 16
```
