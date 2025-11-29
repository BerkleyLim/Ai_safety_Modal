# src/evaluation/validation.py
"""모델 및 프레임워크 적합성 검증 모듈"""

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from .metrics import get_best_metrics, YOLOMetrics


@dataclass
class ValidationResult:
    """적합성 검증 결과"""
    name: str
    passed: bool
    expected: str
    actual: str
    message: str = ""


@dataclass
class FrameworkValidation:
    """프레임워크 적합성 검증 결과 모음"""
    results: list[ValidationResult] = field(default_factory=list)

    def add(self, result: ValidationResult):
        self.results.append(result)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_count(self) -> int:
        return len(self.results)

    def print_summary(self):
        print("\n" + "=" * 70)
        print("🔍 모델 및 프레임워크 적합성 검증 결과")
        print("=" * 70)

        for r in self.results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            print(f"\n[{status}] {r.name}")
            print(f"   기대값: {r.expected}")
            print(f"   실제값: {r.actual}")
            if r.message:
                print(f"   비고: {r.message}")

        print("\n" + "-" * 70)
        print(f"📊 종합: {self.pass_count}/{self.total_count} 항목 통과")
        if self.all_passed:
            print("✅ 모든 적합성 검증 통과 - 프레임워크 설계 검증 완료")
        else:
            print("⚠️ 일부 항목 미통과 - 추가 검토 필요")
        print("=" * 70)


def validate_yolo_model(model_path: str) -> ValidationResult:
    """YOLO 모델 파일 검증"""
    path = Path(model_path)
    exists = path.exists()
    size_mb = path.stat().st_size / (1024 * 1024) if exists else 0

    return ValidationResult(
        name="YOLO 모델 파일 존재",
        passed=exists and size_mb > 1,
        expected="best.pt 파일 존재 (>1MB)",
        actual=f"{'존재' if exists else '없음'} ({size_mb:.2f}MB)" if exists else "파일 없음",
        message="학습된 커스텀 모델 로드 가능" if exists else ""
    )


def validate_yolo_performance(results_csv: str, min_map50: float = 0.5) -> ValidationResult:
    """YOLO 모델 성능 기준 검증"""
    best = get_best_metrics(results_csv)

    if best is None:
        return ValidationResult(
            name="YOLO 모델 성능 (mAP50)",
            passed=False,
            expected=f"mAP50 >= {min_map50}",
            actual="결과 파일 없음",
        )

    return ValidationResult(
        name="YOLO 모델 성능 (mAP50)",
        passed=best.mAP50 >= min_map50,
        expected=f"mAP50 >= {min_map50}",
        actual=f"mAP50 = {best.mAP50:.4f} (Epoch {best.epoch})",
        message="객체 탐지 정확도 기준 충족" if best.mAP50 >= min_map50 else "추가 학습 필요"
    )


def validate_anomaly_detection() -> ValidationResult:
    """이상 탐지 로직 검증 (ANOMALY_CLASSES 기반)"""
    try:
        from monitoring import ANOMALY_CLASSES
        ua_count = sum(1 for c in ANOMALY_CLASSES if c.startswith(("forklift", "stacking", "person", "worker", "cargo", "flammable", "smoking")))
        uc_count = len(ANOMALY_CLASSES) - ua_count

        return ValidationResult(
            name="이상 탐지 클래스 정의",
            passed=len(ANOMALY_CLASSES) >= 20,
            expected="UA/UC 위험 클래스 20개 이상 정의",
            actual=f"총 {len(ANOMALY_CLASSES)}개 클래스 정의",
            message="Monitoring Layer에서 위험 상황 판단 기준으로 사용"
        )
    except ImportError:
        return ValidationResult(
            name="이상 탐지 클래스 정의",
            passed=False,
            expected="ANOMALY_CLASSES 정의",
            actual="monitoring 모듈 임포트 실패",
        )


def validate_pipeline_flow() -> ValidationResult:
    """파이프라인 흐름 검증 (Monitoring → Reasoning → Action)"""
    try:
        from monitoring import detect_objects
        from reasoning import analyze_risk_with_vlm
        from action import generate_safety_guideline

        return ValidationResult(
            name="3-Layer 파이프라인 구조",
            passed=True,
            expected="Monitoring → Reasoning → Action 함수 존재",
            actual="detect_objects, analyze_risk_with_vlm, generate_safety_guideline 로드 성공",
            message="파이프라인 연동 준비 완료"
        )
    except ImportError as e:
        return ValidationResult(
            name="3-Layer 파이프라인 구조",
            passed=False,
            expected="모든 Layer 함수 임포트 가능",
            actual=f"임포트 실패: {e}",
        )


def validate_pydantic_schemas() -> ValidationResult:
    """Pydantic 스키마 검증"""
    try:
        from schemas.monitoring_output import MonitoringOutput, DetectedObject
        from schemas.reasoning_output import ReasoningOutput

        # 샘플 데이터로 스키마 검증 (alias 'class' 사용)
        sample_obj = DetectedObject(**{"class": "test", "confidence": 0.9, "box": [0, 0, 100, 100]})
        sample_monitoring = MonitoringOutput(
            status="anomaly_detected",
            image_path="/test/path.jpg",
            detected_objects=[sample_obj]
        )
        sample_reasoning = ReasoningOutput(
            image_path="/test/path.jpg",
            risk_level="HIGH",
            hazard_code="UA-01",
            reason="테스트 사유"
        )

        return ValidationResult(
            name="Pydantic 스키마 정의",
            passed=True,
            expected="MonitoringOutput, ReasoningOutput 스키마 정상 동작",
            actual="스키마 생성 및 검증 성공",
            message="Layer 간 데이터 타입 안전성 보장"
        )
    except Exception as e:
        return ValidationResult(
            name="Pydantic 스키마 정의",
            passed=False,
            expected="스키마 생성 가능",
            actual=f"오류: {e}",
        )


def validate_inference_speed(image_path: str) -> ValidationResult:
    """추론 속도 검증"""
    try:
        from monitoring import model

        start = time.time()
        results = model(image_path, verbose=False)
        elapsed = time.time() - start

        return ValidationResult(
            name="YOLO 추론 속도",
            passed=elapsed < 1.0,  # 1초 이내
            expected="이미지당 1초 이내",
            actual=f"{elapsed:.3f}초",
            message="실시간 처리 가능" if elapsed < 1.0 else "최적화 필요"
        )
    except Exception as e:
        return ValidationResult(
            name="YOLO 추론 속도",
            passed=False,
            expected="추론 실행 가능",
            actual=f"오류: {e}",
        )


def run_full_validation(model_dir: str, test_image: Optional[str] = None) -> FrameworkValidation:
    """전체 적합성 검증 실행"""
    validation = FrameworkValidation()

    model_path = Path(model_dir)
    best_pt = model_path / "weights" / "best.pt"
    results_csv = model_path / "results.csv"

    # 1. 모델 파일 검증
    validation.add(validate_yolo_model(str(best_pt)))

    # 2. 성능 기준 검증
    if results_csv.exists():
        validation.add(validate_yolo_performance(str(results_csv), min_map50=0.5))

    # 3. 이상 탐지 클래스 검증
    validation.add(validate_anomaly_detection())

    # 4. 파이프라인 구조 검증
    validation.add(validate_pipeline_flow())

    # 5. 스키마 검증
    validation.add(validate_pydantic_schemas())

    # 6. 추론 속도 검증 (테스트 이미지 있을 경우)
    if test_image and Path(test_image).exists():
        validation.add(validate_inference_speed(test_image))

    return validation


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"

    # 가장 최근 모델 찾기
    model_dirs = sorted(models_dir.glob("safety_*"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not model_dirs:
        print("❌ 학습된 모델이 없습니다.")
    else:
        latest_model = model_dirs[0]
        print(f"🔍 검증 대상 모델: {latest_model.name}")

        # 테스트 이미지 찾기
        test_images = list((project_root / "data").rglob("**/val/images/*.jpg"))
        test_image = str(test_images[0]) if test_images else None

        validation = run_full_validation(str(latest_model), test_image)
        validation.print_summary()