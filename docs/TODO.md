# 프로젝트 진행 TODO

## 전체 진행 현황

| 단계 | 상태 | 설명 |
|------|------|------|
| 데이터 전처리 | ✅ 완료 | AI Hub → YOLO 형식 변환 |
| 모델 구현 | ✅ 완료 | 3개 Layer 구현 + 파이프라인 연동 |
| 실험 설계 | ⬜ 예정 | 평가 지표 및 시나리오 정의 |
| 실험 수행 | 🔄 진행중 | 모델 학습 및 파이프라인 테스트 |
| 성능 검증 | ⬜ 예정 | 결과 분석 및 정리 |

---

## 1. 데이터 전처리 ✅

### 수행 내용
- [x] AI Hub 물류센터 안전 데이터 다운로드
- [x] 전처리 스크립트 작성 (`preprocessing/aihub_to_yolo.py`)
- [x] 11개 카테고리 YOLO 형식 변환

### 전처리 상세

**입력 (AI Hub 원본)**
```
data/ai_hub/
├── traning/
│   ├── original/    # 이미지 (TS_01~11)
│   └── label/       # JSON 라벨 (TL_01~11)
└── validation/
    ├── original/    # 이미지 (VS_01~11)
    └── label/       # JSON 라벨 (VL_01~11)
```

**출력 (YOLO 형식)**
```
data/
├── 01_도크설비/logistics_yolo/
│   ├── data.yaml
│   ├── train/images/, train/labels/
│   └── val/images/, val/labels/
├── 02_보관/logistics_yolo/
└── ... (11개 카테고리)
```

**변환 작업**
- JSON 라벨 파싱 (bbox, polygon 처리)
- 좌표 변환: 픽셀 좌표 → YOLO 정규화 좌표 (0~1)
- 클래스 매핑: AI Hub class_id (SO-01, WO-01, ...) → YOLO class index (0~56)
- 57개 클래스 정의 (정적객체 21 + 동적객체 8 + 위험행동 13 + 위험상태 15)

---

## 2. 모델 구현 ✅

### Monitoring Layer - YOLO 학습 (이상탐지)
- [x] 학습 스크립트 작성 (`training/train_yolo.py`)
- [x] 추론 코드 작성 (`monitoring/__init__.py`)
- [x] 위험 클래스 정의 (ANOMALY_CLASSES: UA, UC 클래스)
- [x] dict→Pydantic 변환 처리 (`run.py` 수정)
- [ ] 11개 카테고리 모델 학습 실행 (용량 제약으로 일부만 수행)

### Reasoning Layer - VLM 위험분류, 근거생성
- [x] GPT-4o Vision API 연동 (`reasoning/__init__.py`)
- [x] 프롬프트 작성 (위험 수준 판정 + 근거 생성)
- [x] 출력 스키마 정의 (risk_level: LOW/MED/HIGH, reason)
- [x] 이미지 Base64 인코딩 처리

### Action Layer - LLM 안전지침 다국어 생성
- [x] GPT-4o API 연동 (`action/__init__.py`)
- [x] 다국어 생성 (한국어, 영어, 베트남어)
- [x] 위험등급별 분기 처리
  - LOW: 로그 기록
  - MED: 관리자 확인 요청
  - HIGH: 다국어 가이드라인 생성

---

## 3. 실험 설계 ⬜

### 평가 지표 정의
- [ ] YOLO 모델 평가
  - mAP50: IoU 0.5 기준 평균 정밀도
  - mAP50-95: IoU 0.5~0.95 기준 평균 정밀도
  - Precision: 정밀도
  - Recall: 재현율
- [ ] VLM 평가
  - 위험 분류 정확도 (LOW/MED/HIGH)
  - 근거 생성 적절성 (정성 평가)
- [ ] 전체 파이프라인 평가
  - 처리 속도 (이미지당 소요 시간)
  - End-to-End 정확도

### 테스트 데이터셋 구성
- [ ] 카테고리별 validation 이미지 선정
- [ ] 위험 상황별 샘플 이미지 선정
  - 정상 상황 샘플
  - 위험 행동 (UA) 샘플
  - 위험 상태 (UC) 샘플

### 실험 시나리오 작성
- [ ] 시나리오 1: 정상 상황 → 탐지 없음 확인
- [ ] 시나리오 2: 위험 행동 (UA) → 탐지 + 분석 + 가이드라인
- [ ] 시나리오 3: 위험 상태 (UC) → 탐지 + 분석 + 가이드라인

---

## 4. 실험 수행 ⬜

### YOLO 모델 학습
```bash
cd src
python -m training.train_yolo --category 01 --epochs 100
```

- [ ] 카테고리 01 (도크설비)
- [ ] 카테고리 02 (보관)
- [ ] 카테고리 03 (부가가치서비스)
- [ ] 카테고리 04 (설비및장비)
- [ ] 카테고리 05 (운반)
- [x] 카테고리 06 (입고) ✅ 완료 (샘플 100개, 100 epochs)
  - mAP50: 0.7598 | mAP50-95: 0.5481 | Precision: 0.7933 | Recall: 0.6539
  - 추론 속도: 0.075초/이미지
- [ ] 카테고리 07 (지게차)
- [ ] 카테고리 08 (출고)
- [ ] 카테고리 09 (파렛트렉)
- [ ] 카테고리 10 (피킹분배)
- [ ] 카테고리 11 (화재)

### 파이프라인 테스트
```bash
cd src
python run.py --image <이미지경로>
```

- [x] Monitoring → Reasoning → Action 연동 확인 ✅ (API quota 제외 정상 동작)
- [ ] 각 시나리오별 테스트 수행

### 파이프라인 테스트 결과 (2024-11-29)
- **테스트 이미지**: `06_입고/val/images/image_000001.jpg`
- **Monitoring 결과**:
  - 탐지된 객체: cargo_truck, stacked_cargo_group, forklift_path, worker_uniform, forklift, **worker_in_truck_loading (UC-02)**
  - 이상 상황 판단: ✅ `worker_in_truck_loading` 탐지 → Reasoning Layer로 전달
- **Reasoning 결과**: OpenAI API quota 초과 (429 에러) - 코드 구조는 정상

---

## 5. 성능 검증 ⬜

### YOLO 모델 성능 측정
- [ ] 카테고리별 mAP, Precision, Recall 기록
- [ ] 학습 곡선 분석 (loss 변화, metrics 변화)
- [ ] Confusion Matrix 분석

### VLM 분석 품질 평가
- [ ] 위험 분류 정확도 측정
- [ ] 근거 생성 적절성 평가
- [ ] 오분류 케이스 분석

### 전체 시스템 평가
- [ ] 처리 속도 측정 (이미지당 평균 소요 시간)
- [ ] 오탐(False Positive) / 미탐(False Negative) 분석
- [ ] 실제 위험 상황 대응 적절성 평가

### 결과 정리
- [ ] 실험 결과 표/그래프 작성
- [ ] 결과 분석 및 개선점 도출
- [ ] 최종 보고서 작성

---

## 6. 평가 도구 ✅

### 구현 완료 (`src/evaluation/`)
- [x] `metrics.py`: 성능 지표 추출 (mAP50, Precision, Recall 등)
- [x] `validation.py`: 프레임워크 적합성 검증 (6개 항목)
- [x] `visualize.py`: 학습 결과 시각화 (대시보드 생성)
- [x] `generate_report.py`: 통합 리포트 CLI

### 실행 방법
```bash
cd src
python -m evaluation.generate_report              # 전체 리포트
python -m evaluation.generate_report --action metrics   # 성능 지표만
python -m evaluation.generate_report --action validate  # 적합성 검증만
python -m evaluation.generate_report --action visualize # 시각화만
```

### 적합성 검증 결과 (06_입고 모델)
| 항목 | 결과 |
|------|------|
| YOLO 모델 파일 존재 | ✅ PASS (5.98MB) |
| mAP50 >= 0.5 | ✅ PASS (0.7598) |
| ANOMALY_CLASSES 정의 | ✅ PASS (28개) |
| 3-Layer 파이프라인 | ✅ PASS |
| Pydantic 스키마 | ✅ PASS |
| 추론 속도 < 1초 | ✅ PASS (0.075초) |