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

# 카테고리 06~11 전처리 (샘플 500개씩)
python -m preprocessing.aihub_to_yolo --folders 06 07 08 09 10 11 --sample 500

# 전체 카테고리 전처리 (전체 데이터를 말함)
python -m preprocessing.aihub_to_yolo --folders 01 02 03 04 05 06 07 08 09 10 11
```

### 2. 학습 (YOLO 모델)

```bash
cd src

# 카테고리별 학습
python -m training.train_yolo --category 01 --epochs 100  # 도크설비
python -m training.train_yolo --category 02 --epochs 100  # 보관
python -m training.train_yolo --category 03 --epochs 100  # 부가가치서비스
python -m training.train_yolo --category 04 --epochs 100  # 설비및장비
python -m training.train_yolo --category 05 --epochs 100  # 운반
python -m training.train_yolo --category 06 --epochs 100  # 입고 (데이터 다운로드 후)
python -m training.train_yolo --category 07 --epochs 100  # 지게차 (데이터 다운로드 후)
python -m training.train_yolo --category 08 --epochs 100  # 출고 (데이터 다운로드 후)
python -m training.train_yolo --category 09 --epochs 100  # 파렛트렉 (데이터 다운로드 후)
python -m training.train_yolo --category 10 --epochs 100  # 피킹분배 (데이터 다운로드 후)
python -m training.train_yolo --category 11 --epochs 100  # 화재 (데이터 다운로드 후)

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
python run.py --image ../data/06_입고/logistics_yolo/val/images/image_000001.jpg

# 이미지 자동 선택 (전처리된 val 이미지 중 하나)
python run.py
```

### 4. 평가 및 검증

```bash
cd src

# 전체 리포트 생성 (성능 지표 + 적합성 검증 + 시각화)
python -m evaluation.generate_report

# 성능 지표만 출력
python -m evaluation.generate_report --action metrics

# 적합성 검증만 실행
python -m evaluation.generate_report --action validate

# 시각화만 생성
python -m evaluation.generate_report --action visualize

# 특정 모델 지정
python -m evaluation.generate_report --model ../models/safety_06_입고_20251129_134939
```

**평가 모듈 설명:**
- `evaluation.metrics`: YOLO 학습 결과(mAP50, Precision, Recall 등) 추출 및 출력
- `evaluation.validation`: 모델 및 프레임워크 적합성 검증 (6개 항목)
- `evaluation.visualize`: 학습 곡선 및 성능 대시보드 시각화

**적합성 검증 항목:**
1. YOLO 모델 파일 존재 여부
2. YOLO 모델 성능 기준 (mAP50 >= 0.5)
3. 이상 탐지 클래스 정의 (ANOMALY_CLASSES)
4. 3-Layer 파이프라인 구조 (Monitoring → Reasoning → Action)
5. Pydantic 스키마 정의
6. YOLO 추론 속도 (1초 이내)

## 프로젝트 구조

```
src/
├── preprocessing/          # 전처리 모듈
│   └── aihub_to_yolo.py   # AI Hub -> YOLO 변환
├── training/              # 학습 모듈
│   └── train_yolo.py      # YOLO 학습
├── monitoring/            # 객체 탐지 (YOLO)
├── reasoning/             # 위험 분석 (VLM)
├── action/                # 안전 가이드라인 생성 (LLM)
├── evaluation/            # 평가 및 검증 모듈
│   ├── metrics.py         # 성능 지표 추출
│   ├── validation.py      # 적합성 검증
│   └── visualize.py       # 시각화
├── schemas/               # Pydantic 스키마
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
| 06 | 입고 |
| 07 | 지게차 |
| 08 | 출고 |
| 09 | 파렛트렉 |
| 10 | 피킹분배 |
| 11 | 화재 |

## 데이터셋

### 출처

- **AI Hub**: [물류센터 안전장비 및 행동 인식 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=510)

### 다운로드 후 셋팅

압축 해제된 상태에서 진행합니다.

1. 프로젝트 루트에 `data/ai_hub/` 폴더 생성
2. 다운로드한 데이터를 아래 구조로 배치:

```
data/ai_hub/
├── traning/
│   ├── original/
│   │   ├── TS_01_도크설비/
│   │   ├── TS_02_보관/
│   │   ├── TS_03_부가가치서비스/
│   │   ├── TS_04_설비 및 장비/
│   │   ├── TS_05_운반/
│   │   ├── TS_06_입고/
│   │   ├── TS_07_지게차/
│   │   ├── TS_08_출고/
│   │   ├── TS_09_파렛트, 렉/
│   │   ├── TS_10_피킹, 분배/
│   │   └── TS_11_화재/
│   └── label/
│       ├── TL_01_도크설비/
│       ├── TL_02_보관/
│       └── ... (TL_03 ~ TL_11)
└── validation/
    ├── original/
    │   ├── VS_01_도크설비/
    │   └── ... (VS_02 ~ VS_11)
    └── label/
        ├── VL_01_도크설비/
        └── ... (VL_02 ~ VL_11)
```

3. 전처리 실행:
```bash
cd src
python -m preprocessing.aihub_to_yolo --folders 01 02 03 04 05 06 07 08 09 10 11 --sample 500
```

## 클래스 (57개)

- **정적 객체 (SO) 21개**: 보관랙, 적재물류, 도크, 출입문, 화물승강기, 멀티탭, 소화기, 안전구역, 지게차이동영역, 안전펜스 등
- **동적 객체 (WO) 8개**: 작업자(작업복착용/미착용), 화물트럭, 지게차, 핸드파레트카, 롤테이너, 운반수레, 흡연
- **위험 행동 (UA) 13개**: 지게차 시야미확보, 화물 붕괴, 평치적재, 적재불량, 지게차 통로 내 사람, 용접구역 가연물 등
- **위험 상태 (UC) 15개**: 트럭 내 작업자, 지게차 통로 미표시, 도크 장애물, 파렛트 파손, 소화기 미비치, 대피로 적재물 등