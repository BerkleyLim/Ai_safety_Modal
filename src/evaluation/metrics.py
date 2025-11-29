# src/evaluation/metrics.py
"""YOLO 모델 성능 지표 추출 및 출력 모듈"""

import csv
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class YOLOMetrics:
    """YOLO 학습 결과 성능 지표"""
    epoch: int
    precision: float
    recall: float
    mAP50: float
    mAP50_95: float
    train_box_loss: float
    train_cls_loss: float
    val_box_loss: float
    val_cls_loss: float

    def __str__(self):
        return (
            f"Epoch {self.epoch} 성능 지표:\n"
            f"  - Precision: {self.precision:.4f}\n"
            f"  - Recall: {self.recall:.4f}\n"
            f"  - mAP50: {self.mAP50:.4f}\n"
            f"  - mAP50-95: {self.mAP50_95:.4f}\n"
            f"  - Train Box Loss: {self.train_box_loss:.4f}\n"
            f"  - Train Cls Loss: {self.train_cls_loss:.4f}\n"
            f"  - Val Box Loss: {self.val_box_loss:.4f}\n"
            f"  - Val Cls Loss: {self.val_cls_loss:.4f}"
        )

    def to_dict(self):
        return {
            "epoch": self.epoch,
            "precision": self.precision,
            "recall": self.recall,
            "mAP50": self.mAP50,
            "mAP50_95": self.mAP50_95,
            "train_box_loss": self.train_box_loss,
            "train_cls_loss": self.train_cls_loss,
            "val_box_loss": self.val_box_loss,
            "val_cls_loss": self.val_cls_loss,
        }


def load_yolo_metrics(results_csv_path: str) -> list[YOLOMetrics]:
    """results.csv에서 모든 에포크의 성능 지표 로드"""
    metrics_list = []

    with open(results_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 컬럼명에 공백이 있을 수 있으므로 strip 처리
            row = {k.strip(): v for k, v in row.items()}

            metrics = YOLOMetrics(
                epoch=int(row['epoch']),
                precision=float(row['metrics/precision(B)']),
                recall=float(row['metrics/recall(B)']),
                mAP50=float(row['metrics/mAP50(B)']),
                mAP50_95=float(row['metrics/mAP50-95(B)']),
                train_box_loss=float(row['train/box_loss']),
                train_cls_loss=float(row['train/cls_loss']),
                val_box_loss=float(row['val/box_loss']),
                val_cls_loss=float(row['val/cls_loss']),
            )
            metrics_list.append(metrics)

    return metrics_list


def get_best_metrics(results_csv_path: str) -> Optional[YOLOMetrics]:
    """mAP50 기준 최고 성능 에포크의 지표 반환"""
    metrics_list = load_yolo_metrics(results_csv_path)
    if not metrics_list:
        return None

    return max(metrics_list, key=lambda m: m.mAP50)


def get_final_metrics(results_csv_path: str) -> Optional[YOLOMetrics]:
    """마지막 에포크의 성능 지표 반환"""
    metrics_list = load_yolo_metrics(results_csv_path)
    if not metrics_list:
        return None

    return metrics_list[-1]


def find_model_results(models_dir: str) -> list[dict]:
    """models/ 폴더에서 모든 학습 결과 검색"""
    models_path = Path(models_dir)
    results = []

    for model_dir in models_path.iterdir():
        if model_dir.is_dir() and model_dir.name.startswith("safety_"):
            results_csv = model_dir / "results.csv"
            if results_csv.exists():
                best = get_best_metrics(str(results_csv))
                final = get_final_metrics(str(results_csv))

                results.append({
                    "model_name": model_dir.name,
                    "model_path": str(model_dir / "weights" / "best.pt"),
                    "best_metrics": best,
                    "final_metrics": final,
                })

    return results


def print_model_summary(models_dir: str):
    """모든 학습된 모델의 성능 요약 출력"""
    results = find_model_results(models_dir)

    if not results:
        print("❌ 학습된 모델이 없습니다.")
        return

    print("\n" + "=" * 60)
    print("📊 YOLO 모델 성능 요약")
    print("=" * 60)

    for r in results:
        print(f"\n🔹 {r['model_name']}")
        print(f"   모델 경로: {r['model_path']}")

        if r['best_metrics']:
            m = r['best_metrics']
            print(f"   [Best - Epoch {m.epoch}]")
            print(f"   mAP50: {m.mAP50:.4f} | mAP50-95: {m.mAP50_95:.4f}")
            print(f"   Precision: {m.precision:.4f} | Recall: {m.recall:.4f}")

        if r['final_metrics']:
            m = r['final_metrics']
            print(f"   [Final - Epoch {m.epoch}]")
            print(f"   mAP50: {m.mAP50:.4f} | mAP50-95: {m.mAP50_95:.4f}")
            print(f"   Precision: {m.precision:.4f} | Recall: {m.recall:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys

    # 프로젝트 루트 기준 models 폴더
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"

    print_model_summary(str(models_dir))