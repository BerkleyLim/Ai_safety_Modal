# 데이터 전처리 모듈

물류창고 안전 데이터셋(AI Hub)의 JSON 라벨링 데이터와 이미지를 전처리하는 모듈입니다.

## 📁 파일 구조

```
preprocessing/
├── README.md                   # 이 문서
├── __init__.py                 # 모듈 초기화 및 주요 함수 export
├── data_loader.py              # 데이터 로딩 (JSON, 이미지)
├── data_validator.py           # 데이터 유효성 검증
├── data_augmentation.py        # 데이터 증강 (회전, 반전, 밝기 등)
├── pipeline.py                 # 전체 전처리 파이프라인 통합
└── example_usage.py            # 사용 예시 및 실행 스크립트
```

---

## 📄 각 파일 설명

### 1. `__init__.py`
**역할**: 모듈 초기화 및 주요 함수 export

**내용**:
- 주요 함수들을 모듈 레벨에서 import 가능하게 설정
- `load_json_labels`, `load_image`, `validate_dataset`, `augment_image`, `preprocess_pipeline` export

**사용 예시**:
```python
from preprocessing import load_json_labels, preprocess_pipeline
```

---

### 2. `data_loader.py`
**역할**: JSON 라벨 파일과 이미지 파일을 로드하는 유틸리티

**주요 함수**:
- `load_json_labels(json_path)`: JSON 라벨 파일 로드
- `load_image(image_path, mode='PIL')`: 이미지 로드 (PIL 또는 cv2)
- `parse_annotations(json_data)`: JSON에서 어노테이션(bbox, class) 추출
- `get_image_info(json_data)`: JSON에서 이미지 메타정보 추출
- `scan_dataset(data_dir, ext='.json')`: 디렉토리에서 모든 JSON 파일 스캔

**특징**:
- AI Hub 데이터셋의 JSON 구조에 맞게 파싱
- COCO 형식 및 Custom 형식 지원
- PIL/OpenCV 양쪽 이미지 로딩 지원

**사용 예시**:
```python
from preprocessing.data_loader import load_json_labels, load_image

json_data = load_json_labels("path/to/label.json")
image = load_image("path/to/image.jpg", mode='PIL')
annotations = parse_annotations(json_data)
```

---

### 3. `data_validator.py`
**역할**: 데이터의 무결성 및 유효성을 검증

**주요 클래스/함수**:
- `DataValidator`: 데이터 검증 클래스
  - `validate_image(image_path)`: 이미지 파일 존재, 손상 여부, 크기 확인
  - `validate_bbox(bbox, width, height)`: bbox가 이미지 범위 내에 있는지, 최소 크기 충족하는지 확인
  - `validate_annotation(annotation, width, height)`: 어노테이션 필수 필드 및 bbox 검증
  - `validate_dataset(json_data_list, image_dir)`: 전체 데이터셋 일괄 검증
  - `print_summary()`: 검증 결과 요약 출력

**특징**:
- 최소 bbox 크기 설정 가능 (기본 10픽셀)
- 에러(errors)와 경고(warnings) 구분
- 이미지 손상 감지 (`PIL.Image.verify()`)
- bbox가 이미지 범위를 벗어나는지 확인

**사용 예시**:
```python
from preprocessing.data_validator import DataValidator

validator = DataValidator(min_bbox_size=10)
is_valid = validator.validate_image("path/to/image.jpg")
is_bbox_valid = validator.validate_bbox([100, 100, 200, 150], 1920, 1080)
validator.print_summary()
```

---

### 4. `data_augmentation.py`
**역할**: 이미지와 bbox를 함께 증강하여 학습 데이터 확장

**주요 클래스/함수**:
- `ImageAugmenter`: 데이터 증강 클래스
  - `rotate_image_and_bbox()`: 이미지와 bbox를 함께 회전
  - `flip_image_and_bbox()`: 좌우/상하 반전
  - `adjust_brightness()`: 밝기 조절
  - `adjust_contrast()`: 대비 조절
  - `add_blur()`: 가우시안 블러 추가
  - `augment()`: 위 기법들을 랜덤 또는 전체 적용

**증강 기법**:
1. **회전 (Rotation)**: ±15도 범위 내 랜덤 회전
2. **좌우 반전 (Horizontal Flip)**: 50% 확률
3. **상하 반전 (Vertical Flip)**: 옵션 (기본 비활성화)
4. **밝기 조절 (Brightness)**: 0.8~1.2배 범위
5. **대비 조절 (Contrast)**: 0.8~1.2배 범위
6. **블러 (Blur)**: 10% 확률로 가우시안 블러

**특징**:
- bbox 좌표도 함께 변환 (기하학적 변환)
- 회전 시 bbox의 4개 코너를 모두 회전시킨 후 새 bbox 계산
- 이미지 범위를 벗어나는 bbox는 자동 제거
- 원본 이미지도 결과에 포함

**사용 예시**:
```python
from preprocessing.data_augmentation import ImageAugmenter

augmenter = ImageAugmenter()
bboxes = [[100, 100, 200, 150]]
augmented = augmenter.augment(image, bboxes, augment_all=False)

for aug_img, aug_bboxes, method in augmented:
    print(f"Method: {method}, Objects: {len(aug_bboxes)}")
```

---

### 5. `pipeline.py`
**역할**: 전체 전처리 과정을 통합 관리

**주요 클래스/함수**:
- `PreprocessingPipeline`: 전처리 파이프라인 클래스
  - `__init__()`: 입출력 경로, 증강 옵션, 분할 비율 설정
  - `split_dataset()`: train/val/test 분할 (70%/15%/15% 기본)
  - `process_single_data()`: 단일 데이터 처리 (로드 → 검증 → 증강 → 저장)
  - `run()`: 전체 파이프라인 실행

**파이프라인 단계**:
1. **스캔**: 입력 디렉토리에서 JSON 파일 스캔
2. **분할**: train/val/test로 랜덤 분할
3. **처리**: 각 데이터에 대해
   - JSON 로드
   - 이미지 로드 및 검증
   - bbox 검증
   - 데이터 증강 (train만)
   - 저장 (images/, labels/)
4. **검증 결과 출력**: 성공/실패 통계

**출력 구조**:
```
output_dir/
├── train/
│   ├── images/
│   │   ├── train_000000_0.jpg  # 원본
│   │   ├── train_000000_1.jpg  # 증강1
│   │   └── ...
│   └── labels/
│       ├── train_000000_0.json
│       ├── train_000000_1.json
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**사용 예시**:
```python
from preprocessing import preprocess_pipeline

preprocess_pipeline(
    input_dir='data/raw',
    output_dir='data/processed',
    apply_augmentation=True,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    min_bbox_size=10
)
```

---

### 6. `example_usage.py`
**역할**: 전처리 파이프라인 실행 스크립트 및 사용 예시

**주요 함수**:
- `main()`: 전체 전처리 파이프라인 실행
- `test_single_file()`: 단일 파일 테스트용 함수

**실행 모드**:
1. **full**: 전체 데이터셋 전처리
2. **test**: 단일 파일 테스트

**실행 방법**:
```bash
# 전체 전처리 실행
python example_usage.py --mode full

# 단일 파일 테스트
python example_usage.py --mode test
```

**설정 항목**:
- `INPUT_DIR`: 원본 데이터 경로
- `OUTPUT_DIR`: 전처리 결과 저장 경로
- `APPLY_AUGMENTATION`: 데이터 증강 적용 여부
- `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`: 데이터 분할 비율
- `MIN_BBOX_SIZE`: 최소 bbox 크기 (픽셀)

---

## 🚀 빠른 시작

### 1. 전체 전처리 실행

```python
from preprocessing import preprocess_pipeline

preprocess_pipeline(
    input_dir='../data/raw',
    output_dir='../data/processed',
    apply_augmentation=True,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

### 2. 단일 파일 테스트

```python
from preprocessing.data_loader import load_json_labels, load_image
from preprocessing.data_validator import DataValidator
from preprocessing.data_augmentation import ImageAugmenter

# JSON 로드
json_data = load_json_labels("path/to/label.json")

# 이미지 로드
image = load_image("path/to/image.jpg", mode='PIL')

# 검증
validator = DataValidator(min_bbox_size=10)
is_valid = validator.validate_image("path/to/image.jpg")

# 증강
augmenter = ImageAugmenter()
augmented = augmenter.augment(image, bboxes)
```

---

## 🔧 커스터마이징

### 증강 옵션 변경

```python
from preprocessing.data_augmentation import ImageAugmenter

augmenter = ImageAugmenter(
    rotation_range=30,              # 회전 각도 증가
    brightness_range=(0.7, 1.3),    # 밝기 범위 확대
    flip_vertical=True,             # 상하 반전 활성화
    blur_probability=0.2            # 블러 확률 증가
)
```

### 데이터 분할 비율 변경

```python
preprocess_pipeline(
    input_dir='data/raw',
    output_dir='data/processed',
    train_ratio=0.8,    # 80% 학습
    val_ratio=0.1,      # 10% 검증
    test_ratio=0.1      # 10% 테스트
)
```

---

## 📊 데이터셋 구조 (AI Hub)

### JSON 구조
```json
{
  "Raw data Info.": {
    "raw_data_ID": "L-211227_G19_I_UC-11_008",
    "situation_description": "...",
    "resolution": [1920, 1080]
  },
  "Source data Info.": {
    "source_data_ID": "L-211227_G19_I_UC-11_008_0144",
    "file_extension": "jpg"
  },
  "Learning data info.": {
    "annotation": [
      {
        "class_id": "SO-21",
        "type": "box",
        "coord": [x, y, width, height]
      }
    ]
  }
}
```

### bbox 형식
- **좌표계**: `[x, y, width, height]`
- **x, y**: bbox 좌측 상단 좌표
- **width, height**: bbox 너비와 높이

---

## ⚠️ 주의사항

1. **메모리**: 대용량 데이터셋 처리 시 메모리 부족 주의
2. **디스크 공간**: 증강 시 원본의 3~5배 용량 필요
3. **처리 시간**: 전체 데이터셋 전처리에 수 시간 소요 가능
4. **JSON 구조**: 실제 AI Hub 데이터 다운로드 후 JSON 구조 확인 필요

---

## 📝 TODO

- [ ] 실제 AI Hub JSON 구조에 맞게 `parse_annotations()` 수정
- [ ] 이미지-라벨 매칭 로직 검증
- [ ] 전처리 결과 시각화 도구 추가
- [ ] 진행률 표시 개선 (tqdm)
- [ ] 멀티프로세싱 지원

---

## 🐛 문제 해결

### JSON 파일을 찾을 수 없음
- `INPUT_DIR` 경로가 올바른지 확인
- JSON 파일이 실제로 존재하는지 확인

### 이미지를 로드할 수 없음
- 이미지 파일 경로와 JSON의 `file_name` 일치 여부 확인
- 이미지 파일 손상 여부 확인

### bbox가 이미지 범위를 벗어남
- JSON의 좌표 형식 확인 (`[x, y, w, h]` vs `[x1, y1, x2, y2]`)
- 이미지 해상도와 JSON의 해상도 일치 여부 확인