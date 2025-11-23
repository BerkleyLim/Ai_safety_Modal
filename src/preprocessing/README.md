# 데이터 전처리 모듈

AI Hub 물류창고 안전 데이터셋을 YOLO 형식으로 변환하는 모듈입니다.

## 파일 구조

```
preprocessing/
├── README.md           # 이 문서
├── __init__.py         # 모듈 초기화
└── aihub_to_yolo.py    # AI Hub -> YOLO 변환 스크립트
```

---

## 빠른 시작

### 1. 명령줄에서 전처리 실행

```bash
# 프로젝트 루트에서 실행
cd /path/to/VLM_SafetyGudiance

# 전체 카테고리 변환 (01~11)
python -m src.preprocessing.aihub_to_yolo \
    --data-root ./data/ai_hub \
    --output ./data

# 특정 카테고리만 변환 (01, 02, 03)
python -m src.preprocessing.aihub_to_yolo \
    --data-root ./data/ai_hub \
    --output ./data \
    --folders 01 02 03

# 샘플링 모드 (카테고리당 100개만 처리)
python -m src.preprocessing.aihub_to_yolo \
    --data-root ./data/ai_hub \
    --output ./data \
    --sample 100
```

### 2. Python 코드에서 실행

```python
from src.preprocessing import AIHubToYOLOConverter

# 전체 카테고리 변환
converter = AIHubToYOLOConverter(
    data_root='./data/ai_hub',
    output_base='./data',
    target_folders=None,  # None이면 01~11 전체
    sample_size=None      # None이면 전체 처리
)
converter.run()

# 특정 카테고리만 변환
converter = AIHubToYOLOConverter(
    data_root='./data/ai_hub',
    output_base='./data',
    target_folders=['01', '02', '03'],
    sample_size=100  # 카테고리당 100개만
)
converter.run()
```

---

## 입력 데이터 구조 (AI Hub)

```
data/ai_hub/
├── traning/
│   ├── original/
│   │   ├── TS_01_도크설비/
│   │   │   ├── 불안전한 상태(UC)/
│   │   │   │   └── *.jpg
│   │   │   └── 작업상황(WS)/
│   │   │       └── *.jpg
│   │   ├── TS_02_보관/
│   │   └── ...
│   └── label/
│       ├── TL_01_도크설비/
│       │   ├── 불안전한 상태(UC)/
│       │   │   └── *.json
│       │   └── 작업상황(WS)/
│       │       └── *.json
│       ├── TL_02_보관/
│       └── ...
└── validation/
    ├── original/
    │   └── VS_01_도크설비/
    └── label/
        └── VL_01_도크설비/
```

---

## 출력 데이터 구조 (YOLO 형식)

```
data/
├── 01_도크설비/
│   └── logistics_yolo/
│       ├── data.yaml           # YOLO 학습 설정 파일
│       ├── train/
│       │   ├── images/         # 학습 이미지
│       │   │   ├── image_000000.jpg
│       │   │   └── ...
│       │   └── labels/         # 학습 라벨 (YOLO txt 형식)
│       │       ├── image_000000.txt
│       │       └── ...
│       └── val/
│           ├── images/         # 검증 이미지
│           └── labels/         # 검증 라벨
├── 02_보관/
│   └── logistics_yolo/
│       └── ...
├── 03_부가가치서비스/
├── 04_설비및장비/
├── 05_운반/
├── 06_입고/
├── 07_지게차/
├── 08_출고/
├── 09_파렛트렉/
├── 10_피킹분배/
└── 11_화재/
```

---

## YOLO 라벨 형식

각 `.txt` 파일은 다음 형식입니다:

```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: 클래스 인덱스 (0~34)
- `x_center`, `y_center`: 중심점 좌표 (0~1 정규화)
- `width`, `height`: 너비/높이 (0~1 정규화)

예시:
```
17 0.523438 0.456481 0.119010 0.393519
12 0.650521 0.487037 0.355244 0.239259
```

---

## 클래스 목록 (35개)

| ID | Class ID | 클래스명 (영문) | 설명 |
|----|----------|-----------------|------|
| 0 | SO-01 | safety_helmet | 안전모 |
| 1 | SO-02 | safety_shoes | 안전화 |
| 2 | SO-03 | safety_vest | 안전조끼 |
| 3 | SO-06 | floor | 바닥 |
| 4 | SO-07 | safety_sign | 안전표지판 |
| 5 | SO-08 | fire_extinguisher | 소화기 |
| 6 | SO-12 | safety_railing | 안전난간 |
| 7 | SO-13 | safety_belt | 안전벨트 |
| 8 | SO-14 | safety_zone_polygon | 안전구역 (polygon) |
| 9 | SO-15 | safety_zone | 안전구역 |
| 10 | SO-16 | emergency_exit | 비상구 |
| 11 | SO-17 | safety_net | 안전망 |
| 12 | SO-18 | safety_fence | 안전펜스 |
| 13 | SO-19 | sandwich_panel | 샌드위치패널 |
| 14 | SO-21 | safety_line | 안전라인 |
| 15 | SO-22 | safety_door | 안전문 |
| 16 | SO-23 | safety_gloves | 안전장갑 |
| 17 | WO-01 | person | 사람 |
| 18 | WO-02 | forklift | 지게차 |
| 19 | WO-03 | pallet | 파렛트 |
| 20 | WO-04 | rack | 렉/선반 |
| 21 | WO-05 | cargo | 박스/화물 |
| 22 | WO-06 | conveyor | 컨베이어 |
| 23 | WO-07 | handcart | 핸드카트 |
| 24 | UA-01 | no_helmet | 안전모 미착용 |
| 25 | UA-02 | no_safety_shoes | 안전화 미착용 |
| 26 | UA-03 | no_safety_vest | 안전조끼 미착용 |
| 27 | UA-04 | danger_zone_entry | 위험구역 진입 |
| 28 | UA-05 | phone_while_driving | 운전 중 핸드폰 |
| 29 | UA-06 | speeding | 과속 |
| 30 | UA-16 | other_unsafe_action | 기타 불안전 행동 |
| 31 | UC-09 | pathway_obstacle | 통로 장애물 |
| 32 | UC-10 | improper_stacking | 적재 불량 |
| 33 | UC-15 | poor_lighting | 조명 불량 |
| 34 | UC-16 | other_unsafe_condition | 기타 불안전 상태 |

---

## data.yaml 예시

```yaml
path: /absolute/path/to/01_도크설비/logistics_yolo
train: train/images
val: val/images
nc: 35
names:
  - safety_helmet
  - safety_shoes
  - safety_vest
  - floor
  - safety_sign
  # ... (35개 클래스)
```

---

## YOLO 학습 방법

전처리 완료 후 YOLO 학습:

```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov8n.pt')  # 또는 yolov8s.pt, yolov8m.pt

# 학습 실행
model.train(
    data='./data/01_도크설비/logistics_yolo/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # GPU 사용
)
```

---

## 주의사항

1. **디스크 공간**: 전체 변환 시 원본의 약 2배 용량 필요
2. **처리 시간**: 전체 데이터셋 변환에 수십 분~수 시간 소요
3. **이미지 매칭**: label 폴더와 original 폴더의 구조가 일치해야 함
   - `TL_01_xxx` ↔ `TS_01_xxx`
   - `VL_01_xxx` ↔ `VS_01_xxx`

---

## 문제 해결

### 이미지를 찾을 수 없음
- original 폴더에 해당 이미지가 있는지 확인
- 폴더 이름 매칭 확인 (TL↔TS, VL↔VS)

### 클래스가 인식되지 않음
- JSON의 `class_id`가 `CLASS_MAPPING`에 정의되어 있는지 확인
- 새로운 클래스는 `aihub_to_yolo.py`의 `CLASS_MAPPING`에 추가

### Polygon 타입 처리
- Polygon 좌표는 자동으로 bounding box로 변환됨
- 최소/최대 x, y 좌표로 bbox 계산