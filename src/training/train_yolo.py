"""
YOLOv8 학습 스크립트
물류창고 안전 데이터셋으로 객체 탐지 모델을 학습합니다.

사용법:
    # 단일 카테고리 학습
    python -m src.training.train_yolo --category 01

    # 전체 카테고리 통합 학습
    python -m src.training.train_yolo --category all

    # 커스텀 설정
    python -m src.training.train_yolo --category 01 --epochs 100 --batch 16 --imgsz 640
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

# ultralytics 설치 확인
try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics 패키지가 설치되어 있지 않습니다.")
    print("설치: pip install ultralytics")
    sys.exit(1)


# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# 카테고리 정보
CATEGORIES = {
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


class YOLOTrainer:
    """YOLO 모델 학습 클래스"""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        data_root: str = None,
        output_dir: str = None,
        device: str = "auto"
    ):
        """
        Args:
            model_name: 사전학습 모델 (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
            data_root: 데이터 루트 경로
            output_dir: 모델 저장 경로
            device: 학습 장치 ('auto', 'cpu', '0', '0,1' 등)
        """
        self.model_name = model_name
        self.data_root = Path(data_root) if data_root else DATA_ROOT
        self.output_dir = Path(output_dir) if output_dir else MODELS_DIR
        self.device = device

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_data_yaml_path(self, category: str) -> Optional[Path]:
        """카테고리별 data.yaml 경로 반환"""
        if category == "all":
            # 통합 데이터셋용 data.yaml 경로
            return self.data_root / "all_categories" / "logistics_yolo" / "data.yaml"

        category_name = CATEGORIES.get(category, category)
        yaml_path = self.data_root / f"{category}_{category_name}" / "logistics_yolo" / "data.yaml"

        if yaml_path.exists():
            return yaml_path

        # 폴더명 패턴이 다를 수 있으므로 검색
        for folder in self.data_root.iterdir():
            if folder.is_dir() and folder.name.startswith(f"{category}_"):
                yaml_path = folder / "logistics_yolo" / "data.yaml"
                if yaml_path.exists():
                    return yaml_path

        return None

    def train(
        self,
        category: str,
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 640,
        patience: int = 50,
        workers: int = 8,
        pretrained: bool = True,
        resume: bool = False,
        **kwargs
    ) -> Dict:
        """
        YOLO 모델 학습

        Args:
            category: 학습할 카테고리 번호 (01~11) 또는 'all'
            epochs: 학습 에포크 수
            batch: 배치 크기
            imgsz: 이미지 크기
            patience: Early stopping patience
            workers: 데이터 로더 워커 수
            pretrained: 사전학습 가중치 사용 여부
            resume: 이전 학습 재개 여부
            **kwargs: 추가 학습 파라미터

        Returns:
            학습 결과 딕셔너리
        """
        # data.yaml 경로 확인
        data_yaml = self.get_data_yaml_path(category)
        if data_yaml is None or not data_yaml.exists():
            print(f"오류: data.yaml을 찾을 수 없습니다.")
            print(f"카테고리 {category}의 전처리가 완료되었는지 확인하세요.")
            print(f"예상 경로: {self.data_root}/{category}_xxx/logistics_yolo/data.yaml")
            return {"success": False, "error": "data.yaml not found"}

        # 프로젝트명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        category_name = CATEGORIES.get(category, category)
        project_name = f"safety_{category}_{category_name}"

        print("\n" + "=" * 60)
        print(f"YOLO 학습 시작: {project_name}")
        print("=" * 60)
        print(f"모델: {self.model_name}")
        print(f"데이터: {data_yaml}")
        print(f"에포크: {epochs}")
        print(f"배치 크기: {batch}")
        print(f"이미지 크기: {imgsz}")
        print(f"장치: {self.device}")
        print("=" * 60)

        # 모델 로드
        model = YOLO(self.model_name)

        # 학습 실행
        try:
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                patience=patience,
                workers=workers,
                device=self.device if self.device != "auto" else None,
                project=str(self.output_dir),
                name=f"{project_name}_{timestamp}",
                pretrained=pretrained,
                resume=resume,
                verbose=True,
                **kwargs
            )

            # 결과 저장 경로
            save_dir = self.output_dir / f"{project_name}_{timestamp}"

            print("\n" + "=" * 60)
            print("학습 완료!")
            print("=" * 60)
            print(f"모델 저장 위치: {save_dir}")
            print(f"최고 가중치: {save_dir / 'weights' / 'best.pt'}")
            print(f"최종 가중치: {save_dir / 'weights' / 'last.pt'}")

            return {
                "success": True,
                "save_dir": str(save_dir),
                "best_weights": str(save_dir / "weights" / "best.pt"),
                "last_weights": str(save_dir / "weights" / "last.pt"),
                "results": results
            }

        except Exception as e:
            print(f"\n학습 중 오류 발생: {e}")
            return {"success": False, "error": str(e)}

    def validate(self, weights_path: str, data_yaml: str = None) -> Dict:
        """
        학습된 모델 검증

        Args:
            weights_path: 가중치 파일 경로
            data_yaml: 검증 데이터 yaml 경로 (None이면 학습 데이터 사용)

        Returns:
            검증 결과 딕셔너리
        """
        model = YOLO(weights_path)

        if data_yaml:
            results = model.val(data=data_yaml)
        else:
            results = model.val()

        return {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr
        }

    def export(
        self,
        weights_path: str,
        format: str = "onnx",
        imgsz: int = 640
    ) -> str:
        """
        모델 내보내기

        Args:
            weights_path: 가중치 파일 경로
            format: 내보내기 형식 (onnx, torchscript, tflite 등)
            imgsz: 이미지 크기

        Returns:
            내보낸 모델 경로
        """
        model = YOLO(weights_path)
        export_path = model.export(format=format, imgsz=imgsz)
        print(f"모델 내보내기 완료: {export_path}")
        return export_path


def train_yolo(
    category: str = "01",
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    model: str = "yolov8n.pt",
    device: str = "auto",
    **kwargs
) -> Dict:
    """
    YOLO 학습 간편 함수

    Args:
        category: 카테고리 번호 (01~11) 또는 'all'
        epochs: 에포크 수
        batch: 배치 크기
        imgsz: 이미지 크기
        model: 사전학습 모델
        device: 학습 장치
        **kwargs: 추가 파라미터

    Returns:
        학습 결과
    """
    trainer = YOLOTrainer(model_name=model, device=device)
    return trainer.train(
        category=category,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        **kwargs
    )


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv8 물류창고 안전 모델 학습')

    # 필수 인자
    parser.add_argument(
        '--category',
        type=str,
        default='01',
        help='학습할 카테고리 (01~11 또는 all)'
    )

    # 학습 설정
    parser.add_argument('--epochs', type=int, default=100, help='학습 에포크 수')
    parser.add_argument('--batch', type=int, default=16, help='배치 크기')
    parser.add_argument('--imgsz', type=int, default=640, help='이미지 크기')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--workers', type=int, default=8, help='데이터 로더 워커 수')

    # 모델 설정
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='사전학습 모델 선택'
    )

    # 장치 설정
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='학습 장치 (auto, cpu, 0, 0,1 등)'
    )

    # 기타
    parser.add_argument('--resume', action='store_true', help='이전 학습 재개')

    args = parser.parse_args()

    # 카테고리 목록 출력
    print("\n사용 가능한 카테고리:")
    for cat_id, cat_name in CATEGORIES.items():
        print(f"  {cat_id}: {cat_name}")
    print("  all: 전체 통합")

    # 학습 실행
    result = train_yolo(
        category=args.category,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        model=args.model,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        resume=args.resume
    )

    if result["success"]:
        print("\n학습이 성공적으로 완료되었습니다!")
        print(f"모델 위치: {result['best_weights']}")
    else:
        print(f"\n학습 실패: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()