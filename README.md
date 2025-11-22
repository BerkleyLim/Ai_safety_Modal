# VLM_SafetyGudiance

물류창고 안전 관제 시스템 - AI Hub 데이터 기반 YOLO 객체 탐지 + VLM 위험 분석

## 소개

물류창고 내 안전 위험 요소를 자동으로 탐지하고 다국어 안전 지침을 생성하는 시스템입니다.

1. **Monitoring**: YOLOv8 모델로 이미지에서 안전 객체/위험 행동 탐지
2. **Reasoning**: VLM(GPT-4V)으로 탐지 결과 기반 위험 분석
3. **Action**: 분석 결과를 바탕으로 다국어 안전 가이드라인 생성

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 전처리 (AI Hub -> YOLO 형식)

```bash
cd src

# 카테고리 01~05 전처리 (샘플 500개씩)
python -m preprocessing.aihub_to_yolo --folders 01 02 03 04 05 --sample 500

# 전체 데이터 전처리
python -m preprocessing.aihub_to_yolo --folders 01 02 03 04 05
```

### 2. 학습 (YOLO 모델)

```bash
cd src

# 카테고리 01 학습
python -m training.train_yolo --category 01 --epochs 100

# GPU 지정
python -m training.train_yolo --category 01 --epochs 100 --device 0

# 옵션 확인
python -m training.train_yolo --help
```

학습 결과는 `models/` 폴더에 저장됩니다.

### 3. 관제 파이프라인

```bash
cd src

# 전처리된 이미지로 테스트
python run.py --image ../data/01_도크설비/logistics_yolo/val/images/image_000001.jpg

# 이미지 자동 선택 (전처리된 val 이미지 중 하나)
python run.py
```

## 프로젝트 구조

```
src/
├── preprocessing/          # 전처리 모듈
│   └── aihub_to_yolo.py   # AI Hub -> YOLO 변환
├── training/              # 학습 모듈
│   └── train_yolo.py      # YOLO 학습
├── monitoring/            # 객체 탐지 (YOLO)
├── reasoning/             # 위험 분석 (VLM)
├── action/                # 안전 가이드라인 생성
└── run.py                 # 관제 파이프라인 실행
```

## 카테고리

| 번호 | 이름 |
|------|------|
| 01 | 도크설비 |
| 02 | 보관 |
| 03 | 부가가치서비스 |
| 04 | 설비및장비 |
| 05 | 운반 |

## 클래스 (35개)

- **안전 객체 (SO)**: safety_helmet, safety_shoes, safety_vest, floor, safety_sign 등
- **작업 객체 (WO)**: person, forklift, pallet, rack, cargo, conveyor, handcart
- **위험 행동 (UA)**: no_helmet, no_safety_shoes, no_safety_vest, danger_zone_entry 등
- **위험 상태 (UC)**: pathway_obstacle, improper_stacking, poor_lighting 등