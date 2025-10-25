# VLM_SafetyGudiance
Vision-Language Model 기반 제조 현장 안전 관제 시스템

## 프로젝트 개요
제조 공장의 CCTV 영상에서 안전 위험 상황을 자동으로 감지하고, VLM(Vision-Language Model)을 활용하여 위험을 분석한 후, 다국어 안전 지침을 생성하는 AI 시스템입니다.

## 참고 논문
- **제목**: "Improving intelligent perception and decision optimization of pedestrian crossing scenarios in autonomous driving environments through large visual language models"
- 논문에서 제시한 VLM 기반 인지-판단 프레임워크를 제조 현장 안전 관제에 적용

## 시스템 아키텍처

```
비전 데이터 입력 (이미지/영상)
    ↓
[Monitoring Layer] 객체 탐지 (YOLOv8)
    ↓
이상 상황 감지?
    ↓ (Yes)
[Reasoning Layer] 위험 분석 (GPT-4o VLM)
    ↓
[Action Layer] 다국어 안전 지침 생성 (LLM)
    ↓
안전 지침 출력 (한국어/영어)
```

### 3-Layer 구조

1. **Monitoring Layer** (`src/monitoring`)
   - YOLOv8 모델을 사용한 실시간 객체 탐지
   - 안전 규칙 기반 이상 상황 판단
   - 현재: 'person' 객체 감지 시 이상으로 판단 (향후 고도화 예정)

2. **Reasoning Layer** (`src/reasoning`)
   - GPT-4o VLM API를 활용한 이미지 기반 위험 분석
   - 프롬프트 엔지니어링을 통한 정확한 위험 수준 분류 (LOW/MED/HIGH)
   - 위험 원인에 대한 자연어 설명 생성

3. **Action Layer** (`src/action`)
   - 분석 결과를 기반으로 한 다국어 안전 지침 자동 생성
   - 현재: 한국어/영어 지원 (향후 확장 가능)

## 기술 스택

- **객체 탐지**: YOLOv8 (Ultralytics)
- **VLM**: GPT-4o (OpenAI)
- **언어**: Python 3.x
- **주요 라이브러리**:
  - `ultralytics` (YOLO)
  - `openai` (GPT-4o API)
  - `torch` (PyTorch)
  - `python-dotenv` (환경 변수 관리)

## 프로젝트 구조

```
VLM_SafetyGudiance/
├── src/
│   ├── run.py              # 메인 파이프라인
│   ├── monitoring/         # Monitoring Layer
│   │   └── __init__.py     # YOLOv8 객체 탐지
│   ├── reasoning/          # Reasoning Layer
│   │   └── __init__.py     # VLM 위험 분석
│   └── action/             # Action Layer
│       └── __init__.py     # 안전 지침 생성
├── data/
│   └── mock/               # 테스트 이미지
├── yolov8n.pt             # YOLOv8 사전 학습 모델
├── .env                    # API 키 설정 (git에서 제외)
└── README.md
```

## 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요한 패키지 설치
pip install ultralytics openai python-dotenv torch
```

### 2. API 키 설정
프로젝트 루트에 `.env` 파일 생성:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 실행
```bash
cd src
python run.py
```

## 향후 개선 사항

- [ ] Monitoring Layer: 안전모 미착용, 금지 구역 침입 등 세밀한 이상 탐지 규칙 추가
- [ ] Reasoning Layer: 다양한 VLM 모델 비교 및 성능 평가
- [ ] Action Layer: LLM 기반 동적 안전 지침 생성 (현재는 템플릿 기반)
- [ ] 실시간 영상 스트림 처리 기능 추가
- [ ] 웹 대시보드 UI 개발
- [ ] 다국어 지원 확장 (중국어, 베트남어 등)
