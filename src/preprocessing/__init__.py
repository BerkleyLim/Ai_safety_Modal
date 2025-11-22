"""
데이터 전처리 모듈
AI Hub 물류창고 안전 데이터셋을 YOLO 형식으로 변환합니다.

사용법:
    python -m src.preprocessing.aihub_to_yolo --data-root ./data/ai_hub --output ./data

출력 구조:
    data/
    ├── 01_도크설비/logistics_yolo/
    │   ├── data.yaml
    │   ├── train/images/, train/labels/
    │   └── val/images/, val/labels/
    ├── 02_보관/logistics_yolo/
    └── ...
"""

from .aihub_to_yolo import (
    AIHubToYOLOConverter,
    CLASS_MAPPING,
    CLASS_NAMES,
    CATEGORY_NAMES,
    convert_bbox_to_yolo,
    convert_to_yolo_format,
    parse_json_label,
)

__all__ = [
    'AIHubToYOLOConverter',
    'CLASS_MAPPING',
    'CLASS_NAMES',
    'CATEGORY_NAMES',
    'convert_bbox_to_yolo',
    'convert_to_yolo_format',
    'parse_json_label',
]