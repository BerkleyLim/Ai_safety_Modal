# src/evaluation/visualize.py
"""학습 결과 시각화 모듈"""

import csv
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def load_training_history(results_csv: str) -> dict:
    """results.csv에서 학습 히스토리 로드"""
    history = {
        'epoch': [],
        'train_box_loss': [],
        'train_cls_loss': [],
        'val_box_loss': [],
        'val_cls_loss': [],
        'precision': [],
        'recall': [],
        'mAP50': [],
        'mAP50_95': [],
    }

    with open(results_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v for k, v in row.items()}
            history['epoch'].append(int(row['epoch']))
            history['train_box_loss'].append(float(row['train/box_loss']))
            history['train_cls_loss'].append(float(row['train/cls_loss']))
            history['val_box_loss'].append(float(row['val/box_loss']))
            history['val_cls_loss'].append(float(row['val/cls_loss']))
            history['precision'].append(float(row['metrics/precision(B)']))
            history['recall'].append(float(row['metrics/recall(B)']))
            history['mAP50'].append(float(row['metrics/mAP50(B)']))
            history['mAP50_95'].append(float(row['metrics/mAP50-95(B)']))

    return history


def plot_loss_curves(history: dict, save_path: Optional[str] = None, title: str = "학습 손실 곡선"):
    """Loss 곡선 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box Loss
    axes[0].plot(history['epoch'], history['train_box_loss'], label='Train Box Loss', color='blue')
    axes[0].plot(history['epoch'], history['val_box_loss'], label='Val Box Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Box Loss')
    axes[0].set_title('Box Loss 변화')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Classification Loss
    axes[1].plot(history['epoch'], history['train_cls_loss'], label='Train Cls Loss', color='blue')
    axes[1].plot(history['epoch'], history['val_cls_loss'], label='Val Cls Loss', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Classification Loss')
    axes[1].set_title('Classification Loss 변화')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Loss 곡선 저장: {save_path}")

    plt.show()
    return fig


def plot_metrics_curves(history: dict, save_path: Optional[str] = None, title: str = "성능 지표 변화"):
    """성능 지표 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Precision & Recall
    axes[0].plot(history['epoch'], history['precision'], label='Precision', color='green', marker='')
    axes[0].plot(history['epoch'], history['recall'], label='Recall', color='red', marker='')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Precision & Recall')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # mAP
    axes[1].plot(history['epoch'], history['mAP50'], label='mAP50', color='purple', marker='')
    axes[1].plot(history['epoch'], history['mAP50_95'], label='mAP50-95', color='brown', marker='')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mAP')
    axes[1].set_title('mAP 변화')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 성능 지표 그래프 저장: {save_path}")

    plt.show()
    return fig


def plot_combined_dashboard(history: dict, model_name: str, save_path: Optional[str] = None):
    """종합 대시보드 시각화"""
    fig = plt.figure(figsize=(16, 10))

    # 1. Loss 곡선 (왼쪽 상단)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(history['epoch'], history['train_box_loss'], label='Train Box', alpha=0.8)
    ax1.plot(history['epoch'], history['val_box_loss'], label='Val Box', alpha=0.8)
    ax1.plot(history['epoch'], history['train_cls_loss'], label='Train Cls', linestyle='--', alpha=0.8)
    ax1.plot(history['epoch'], history['val_cls_loss'], label='Val Cls', linestyle='--', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('학습 손실 (Loss)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. mAP 곡선 (오른쪽 상단)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(history['epoch'], history['mAP50'], label='mAP50', color='green', linewidth=2)
    ax2.plot(history['epoch'], history['mAP50_95'], label='mAP50-95', color='blue', linewidth=2)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='기준선 (0.5)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP')
    ax2.set_title('평균 정밀도 (mAP)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Best mAP50 표시
    best_idx = history['mAP50'].index(max(history['mAP50']))
    best_epoch = history['epoch'][best_idx]
    best_mAP50 = history['mAP50'][best_idx]
    ax2.scatter([best_epoch], [best_mAP50], color='red', s=100, zorder=5)
    ax2.annotate(f'Best: {best_mAP50:.4f}\n(Epoch {best_epoch})',
                 xy=(best_epoch, best_mAP50),
                 xytext=(best_epoch + 5, best_mAP50 - 0.1),
                 fontsize=10, color='red')

    # 3. Precision & Recall (왼쪽 하단)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(history['epoch'], history['precision'], label='Precision', color='orange', linewidth=2)
    ax3.plot(history['epoch'], history['recall'], label='Recall', color='purple', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('정밀도 & 재현율')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # 4. 최종 성능 요약 (오른쪽 하단)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    # 최종 성능 텍스트
    final_idx = -1
    summary_text = f"""
    📊 {model_name} 학습 결과 요약

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    총 학습 에포크: {history['epoch'][final_idx]}

    [Best 성능 (Epoch {best_epoch})]
    • mAP50: {history['mAP50'][best_idx]:.4f}
    • mAP50-95: {history['mAP50_95'][best_idx]:.4f}
    • Precision: {history['precision'][best_idx]:.4f}
    • Recall: {history['recall'][best_idx]:.4f}

    [Final 성능 (Epoch {history['epoch'][final_idx]})]
    • mAP50: {history['mAP50'][final_idx]:.4f}
    • mAP50-95: {history['mAP50_95'][final_idx]:.4f}
    • Precision: {history['precision'][final_idx]:.4f}
    • Recall: {history['recall'][final_idx]:.4f}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle(f'YOLO 모델 학습 결과 대시보드 - {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 대시보드 저장: {save_path}")

    plt.show()
    return fig


def visualize_model_results(model_dir: str, save_dir: Optional[str] = None):
    """모델 학습 결과 전체 시각화"""
    model_path = Path(model_dir)
    model_name = model_path.name
    results_csv = model_path / "results.csv"

    if not results_csv.exists():
        print(f"❌ results.csv 파일을 찾을 수 없습니다: {results_csv}")
        return

    print(f"📈 {model_name} 학습 결과 시각화 중...")

    # 히스토리 로드
    history = load_training_history(str(results_csv))

    # 저장 경로 설정
    if save_dir:
        save_path = Path(save_dir)
    else:
        save_path = model_path

    # 시각화
    plot_combined_dashboard(
        history,
        model_name,
        save_path=str(save_path / "training_dashboard.png")
    )


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"

    # 가장 최근 모델 찾기
    model_dirs = sorted(models_dir.glob("safety_*"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not model_dirs:
        print("❌ 학습된 모델이 없습니다.")
    else:
        for model_dir in model_dirs:
            visualize_model_results(str(model_dir))