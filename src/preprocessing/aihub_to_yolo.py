"""
AI Hub 물류센터 안전 데이터셋 -> YOLO 형식 변환 스크립트

AI Hub 데이터 구조:
data/ai_hub/
├── traning/
│   ├── original/
│   │   └── TS_01_도크설비/
│   │       ├── 불안전한 상태(UC)/
│   │       └── 작업상황(WS)/
│   └── label/
│       └── TL_01_도크설비/
│           ├── 불안전한 상태(UC)/
│           └── 작업상황(WS)/
└── validation/
    ├── original/
    │   └── VS_01_도크설비/
    └── label/
        └── VL_01_도크설비/

YOLO 출력 구조 (카테고리별 분리):
data/
├── 01_도크설비/
│   └── logistics_yolo/
│       ├── data.yaml
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       └── val/
│           ├── images/
│           └── labels/
├── 02_보관/
│   └── logistics_yolo/
│       ...
└── ...
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import yaml


# 클래스 매핑 정의 (class_id -> 정수 인덱스)
CLASS_MAPPING = {
    # Safety Object (SO) - 안전 관련 객체
    "SO-01": 0,   # 안전모
    "SO-02": 1,   # 안전화
    "SO-03": 2,   # 안전조끼
    "SO-06": 3,   # 바닥
    "SO-07": 4,   # 안전표지판
    "SO-08": 5,   # 소화기
    "SO-12": 6,   # 안전난간
    "SO-13": 7,   # 안전벨트
    "SO-14": 8,   # 안전구역 (polygon)
    "SO-15": 9,   # 안전구역
    "SO-16": 10,  # 비상구
    "SO-17": 11,  # 안전망
    "SO-18": 12,  # 안전펜스
    "SO-19": 13,  # 샌드위치패널
    "SO-21": 14,  # 안전라인
    "SO-22": 15,  # 안전문
    "SO-23": 16,  # 안전장갑

    # Work Object (WO) - 작업 관련 객체
    "WO-01": 17,  # 사람
    "WO-02": 18,  # 지게차
    "WO-03": 19,  # 파렛트
    "WO-04": 20,  # 렉/선반
    "WO-05": 21,  # 박스/화물
    "WO-06": 22,  # 컨베이어
    "WO-07": 23,  # 핸드카트

    # Unsafe Action (UA) - 불안전한 행동
    "UA-01": 24,  # 안전모 미착용
    "UA-02": 25,  # 안전화 미착용
    "UA-03": 26,  # 안전조끼 미착용
    "UA-04": 27,  # 위험구역 진입
    "UA-05": 28,  # 운전 중 핸드폰 사용
    "UA-06": 29,  # 과속
    "UA-16": 30,  # 기타 불안전 행동

    # Unsafe Condition (UC) - 불안전한 상태
    "UC-09": 31,  # 통로 장애물
    "UC-10": 32,  # 적재 불량
    "UC-15": 33,  # 조명 불량
    "UC-16": 34,  # 기타 불안전 상태
}

# 클래스 이름 (YOLO data.yaml에 사용)
CLASS_NAMES = [
    "safety_helmet", "safety_shoes", "safety_vest", "floor", "safety_sign",
    "fire_extinguisher", "safety_railing", "safety_belt", "safety_zone_polygon",
    "safety_zone", "emergency_exit", "safety_net", "safety_fence", "sandwich_panel",
    "safety_line", "safety_door", "safety_gloves",
    "person", "forklift", "pallet", "rack", "cargo", "conveyor", "handcart",
    "no_helmet", "no_safety_shoes", "no_safety_vest", "danger_zone_entry",
    "phone_while_driving", "speeding", "other_unsafe_action",
    "pathway_obstacle", "improper_stacking", "poor_lighting", "other_unsafe_condition"
]

# 카테고리 이름 매핑 (01~11)
CATEGORY_NAMES = {
    "01": "도크설비",
    "02": "보관",
    "03": "부가가치서비스",
    "04": "설비및장비",
    "05": "운반",
    "06": "입고",
    "07": "지게차",
    "08": "출고",
    "09": "파렛트렉",
    "10": "피킹분배",
    "11": "화재"
}


def convert_bbox_to_yolo(
    bbox: List[float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    bbox 좌표를 YOLO 형식으로 변환

    Args:
        bbox: [x, y, width, height] (픽셀 단위)
        img_width: 이미지 너비
        img_height: 이미지 높이

    Returns:
        (x_center, y_center, width, height) 정규화된 값 (0~1)
    """
    x, y, w, h = bbox

    # 중심점 계산
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height

    # 정규화된 너비/높이
    norm_w = w / img_width
    norm_h = h / img_height

    # 범위 제한 (0~1)
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    norm_w = max(0, min(1, norm_w))
    norm_h = max(0, min(1, norm_h))

    return x_center, y_center, norm_w, norm_h


def polygon_to_bbox(polygon: List[List[float]]) -> List[float]:
    """
    Polygon 좌표를 bbox [x, y, w, h]로 변환

    Args:
        polygon: [[x1, y1], [x2, y2], ...]

    Returns:
        [x, y, width, height]
    """
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def parse_json_label(json_path: str) -> Optional[Dict]:
    """
    JSON 라벨 파일 파싱

    Returns:
        {
            'image_id': str,
            'resolution': [width, height],
            'annotations': [{'class_id': str, 'bbox': [x, y, w, h]}, ...]
        }
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Raw data 정보에서 해상도 추출
        raw_info = data.get('Raw data Info.', {})
        resolution = raw_info.get('resolution', [1920, 1080])

        # Source data 정보에서 이미지 ID 추출
        source_info = data.get('Source data Info.', {})
        image_id = source_info.get('source_data_ID', '')

        # Learning data 정보에서 어노테이션 추출
        learning_info = data.get('Learning data info.', {})
        raw_annotations = learning_info.get('annotation', [])

        annotations = []
        for ann in raw_annotations:
            class_id = ann.get('class_id', '')
            ann_type = ann.get('type', 'box')
            coord = ann.get('coord', [])

            if not class_id:
                continue

            # box 타입
            if ann_type == 'box' and len(coord) == 4:
                annotations.append({
                    'class_id': class_id,
                    'bbox': coord
                })
            # polygon 타입 -> bbox로 변환
            elif ann_type == 'polygon' and len(coord) >= 3:
                bbox = polygon_to_bbox(coord)
                annotations.append({
                    'class_id': class_id,
                    'bbox': bbox
                })

        return {
            'image_id': image_id,
            'resolution': resolution,
            'annotations': annotations
        }
    except Exception as e:
        print(f"Error parsing {json_path}: {e}")
        return None


def convert_to_yolo_format(
    annotations: List[Dict],
    img_width: int,
    img_height: int
) -> List[str]:
    """
    어노테이션을 YOLO txt 형식으로 변환

    Returns:
        YOLO 형식 라인 리스트 ["class_id x_center y_center w h", ...]
    """
    lines = []

    for ann in annotations:
        class_id = ann['class_id']
        bbox = ann['bbox']

        # 클래스 매핑 확인
        if class_id not in CLASS_MAPPING:
            continue

        class_idx = CLASS_MAPPING[class_id]

        # bbox 유효성 검사
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue

        # YOLO 형식으로 변환
        x_center, y_center, w, h = convert_bbox_to_yolo(bbox, img_width, img_height)

        # 유효한 bbox만 추가
        if w > 0 and h > 0:
            line = f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            lines.append(line)

    return lines


class AIHubToYOLOConverter:
    """AI Hub 데이터셋을 YOLO 형식으로 변환 (카테고리별 분리)"""

    def __init__(
        self,
        data_root: str,
        output_base: str,
        target_folders: List[str] = None,
        sample_size: int = None
    ):
        """
        Args:
            data_root: AI Hub 데이터 루트 경로 (data/ai_hub 폴더)
            output_base: YOLO 출력 기본 경로 (data/)
            target_folders: 처리할 폴더 번호 리스트 (예: ["01", "02", ...])
            sample_size: 카테고리당 샘플링할 Train 파일 개수 (None이면 전체 처리)
        """
        self.data_root = Path(data_root)
        self.output_base = Path(output_base)
        self.target_folders = target_folders or [f"{i:02d}" for i in range(1, 12)]  # 01~11
        self.sample_size = sample_size

        # 전체 통계
        self.total_stats = {
            'categories_processed': 0,
            'total_train': 0,
            'total_val': 0,
            'class_counts': {}
        }

    def get_category_output_dir(self, folder_num: str) -> Path:
        """카테고리별 출력 디렉토리 경로 반환"""
        category_name = CATEGORY_NAMES.get(folder_num, folder_num)
        return self.output_base / f"{folder_num}_{category_name}" / "logistics_yolo"

    def setup_category_dirs(self, folder_num: str) -> Path:
        """카테고리별 YOLO 폴더 구조 생성"""
        output_dir = self.get_category_output_dir(folder_num)

        for split in ['train', 'val']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        return output_dir

    def find_image_path(self, json_path: Path) -> Optional[Path]:
        """
        JSON 라벨에 대응하는 이미지 파일 경로 찾기

        AI Hub 구조:
        - label: data/ai_hub/traning/label/TL_01_도크설비/불안전한 상태(UC)/xxx.json
        - image: data/ai_hub/traning/original/TS_01_도크설비/불안전한 상태(UC)/xxx.jpg
        """
        json_str = str(json_path)

        # label -> original
        image_str = json_str.replace('/label/', '/original/')

        # TL_ -> TS_, VL_ -> VS_
        image_str = image_str.replace('/TL_', '/TS_')
        image_str = image_str.replace('/VL_', '/VS_')

        # .json -> .jpg
        image_str = image_str.replace('.json', '.jpg')

        image_path = Path(image_str)

        if image_path.exists():
            return image_path

        # jpg가 없으면 png 시도
        image_path_png = Path(image_str.replace('.jpg', '.png'))
        if image_path_png.exists():
            return image_path_png

        return None

    def process_single_file(
        self,
        json_path: Path,
        output_dir: Path,
        split: str,
        index: int,
        stats: Dict
    ) -> bool:
        """
        단일 JSON 파일 처리

        Args:
            json_path: JSON 라벨 파일 경로
            output_dir: 출력 디렉토리
            split: 'train' 또는 'val'
            index: 파일 인덱스
            stats: 통계 딕셔너리

        Returns:
            성공 여부
        """
        # JSON 파싱
        label_data = parse_json_label(str(json_path))
        if not label_data:
            return False

        resolution = label_data['resolution']
        annotations = label_data['annotations']

        if not annotations:
            return False

        # 이미지 파일 찾기
        image_path = self.find_image_path(json_path)
        if not image_path:
            return False

        # 파일명 생성
        safe_filename = f"image_{index:06d}"

        # 이미지 복사
        output_image_path = output_dir / split / 'images' / f"{safe_filename}.jpg"
        shutil.copy2(image_path, output_image_path)

        # YOLO 라벨 생성
        img_width, img_height = resolution
        yolo_lines = convert_to_yolo_format(annotations, img_width, img_height)

        if not yolo_lines:
            output_image_path.unlink()
            return False

        # 라벨 저장
        output_label_path = output_dir / split / 'labels' / f"{safe_filename}.txt"
        with open(output_label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))

        # 클래스 통계 업데이트
        for ann in annotations:
            class_id = ann['class_id']
            if class_id in CLASS_MAPPING:
                stats['class_counts'][class_id] = stats['class_counts'].get(class_id, 0) + 1
                self.total_stats['class_counts'][class_id] = self.total_stats['class_counts'].get(class_id, 0) + 1

        return True

    def collect_category_files(self, split_type: str, folder_num: str) -> List[Path]:
        """
        특정 카테고리의 JSON 파일 목록 수집

        Args:
            split_type: 'traning' 또는 'validation'
            folder_num: 폴더 번호 (01, 02, ...)

        Returns:
            JSON 파일 경로 리스트
        """
        json_files = []
        label_base = self.data_root / split_type / 'label'

        if not label_base.exists():
            return json_files

        # 해당 번호로 시작하는 폴더 찾기
        prefix = f"TL_{folder_num}" if split_type == 'traning' else f"VL_{folder_num}"

        for folder in label_base.iterdir():
            if not folder.is_dir():
                continue

            if folder.name.startswith(prefix):
                for json_file in folder.rglob('*.json'):
                    json_files.append(json_file)

        return json_files

    def create_data_yaml(self, output_dir: Path, folder_num: str, train_count: int, val_count: int):
        """YOLO data.yaml 파일 생성"""
        category_name = CATEGORY_NAMES.get(folder_num, folder_num)

        data = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(CLASS_NAMES),
            'names': CLASS_NAMES
        }

        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    def process_category(self, folder_num: str) -> Dict:
        """단일 카테고리 처리"""
        import random

        category_name = CATEGORY_NAMES.get(folder_num, folder_num)
        print(f"\n{'='*60}")
        print(f"카테고리 {folder_num}_{category_name} 처리 중...")
        print(f"{'='*60}")

        # 통계 초기화
        stats = {
            'train': 0,
            'val': 0,
            'skipped': 0,
            'class_counts': {}
        }

        # 출력 디렉토리 생성
        output_dir = self.setup_category_dirs(folder_num)
        print(f"출력 경로: {output_dir}")

        # Training 데이터 처리
        print(f"\n[Train] 데이터 수집 중...")
        train_files = self.collect_category_files('traning', folder_num)
        print(f"  수집된 JSON: {len(train_files)}개")

        if self.sample_size and len(train_files) > self.sample_size:
            random.shuffle(train_files)
            train_files = train_files[:self.sample_size]
            print(f"  샘플링 후: {len(train_files)}개")

        train_idx = 0
        for json_path in tqdm(train_files, desc="  Train", leave=False):
            if self.process_single_file(json_path, output_dir, 'train', train_idx, stats):
                stats['train'] += 1
                train_idx += 1
            else:
                stats['skipped'] += 1

        # Validation 데이터 처리
        print(f"\n[Val] 데이터 수집 중...")
        val_files = self.collect_category_files('validation', folder_num)
        print(f"  수집된 JSON: {len(val_files)}개")

        val_sample = self.sample_size // 5 if self.sample_size else None
        if val_sample and len(val_files) > val_sample:
            random.shuffle(val_files)
            val_files = val_files[:val_sample]
            print(f"  샘플링 후: {len(val_files)}개")

        val_idx = 0
        for json_path in tqdm(val_files, desc="  Val", leave=False):
            if self.process_single_file(json_path, output_dir, 'val', val_idx, stats):
                stats['val'] += 1
                val_idx += 1
            else:
                stats['skipped'] += 1

        # data.yaml 생성
        self.create_data_yaml(output_dir, folder_num, stats['train'], stats['val'])

        # 결과 출력
        print(f"\n결과:")
        print(f"  Train: {stats['train']}개")
        print(f"  Val: {stats['val']}개")
        print(f"  Skipped: {stats['skipped']}개")

        # 전체 통계 업데이트
        self.total_stats['categories_processed'] += 1
        self.total_stats['total_train'] += stats['train']
        self.total_stats['total_val'] += stats['val']

        return stats

    def run(self):
        """전체 변환 실행"""
        print("\n" + "=" * 60)
        print("AI Hub -> YOLO 변환 시작 (카테고리별 분리)")
        print(f"대상 카테고리: {self.target_folders}")
        if self.sample_size:
            print(f"샘플 모드: 카테고리당 최대 {self.sample_size}개")
        print("=" * 60)

        # 각 카테고리별 처리
        for folder_num in self.target_folders:
            self.process_category(folder_num)

        # 최종 결과 출력
        self.print_total_summary()

    def print_total_summary(self):
        """전체 변환 결과 요약"""
        print("\n" + "=" * 60)
        print("전체 변환 완료!")
        print("=" * 60)
        print(f"\n처리된 카테고리: {self.total_stats['categories_processed']}개")
        print(f"총 Train 이미지: {self.total_stats['total_train']}개")
        print(f"총 Val 이미지: {self.total_stats['total_val']}개")

        print(f"\n클래스별 총 어노테이션 수:")
        for class_id, count in sorted(self.total_stats['class_counts'].items()):
            class_idx = CLASS_MAPPING.get(class_id, -1)
            class_name = CLASS_NAMES[class_idx] if 0 <= class_idx < len(CLASS_NAMES) else "unknown"
            print(f"  {class_id} ({class_name}): {count}")

        print(f"\n출력 구조:")
        for folder_num in self.target_folders:
            category_name = CATEGORY_NAMES.get(folder_num, folder_num)
            print(f"  {self.output_base}/{folder_num}_{category_name}/logistics_yolo/")


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='AI Hub 물류센터 안전 데이터셋을 YOLO 형식으로 변환 (카테고리별)')
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data/ai_hub',
        help='AI Hub 데이터 루트 경로 (default: ./data/ai_hub)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./data',
        help='YOLO 출력 기본 경로 (default: ./data)'
    )
    parser.add_argument(
        '--folders',
        type=str,
        nargs='+',
        default=None,
        help='처리할 폴더 번호 (default: 전체 01~11)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='카테고리당 샘플링할 Train 파일 개수 (미지정 시 전체 처리)'
    )

    args = parser.parse_args()

    converter = AIHubToYOLOConverter(
        data_root=args.data_root,
        output_base=args.output,
        target_folders=args.folders,
        sample_size=args.sample
    )
    converter.run()


if __name__ == "__main__":
    main()