"""
YOLO 모델 학습 모듈
물류창고 안전 데이터셋으로 YOLOv8 모델을 학습합니다.
"""

from .train_yolo import train_yolo, YOLOTrainer

__all__ = ['train_yolo', 'YOLOTrainer']