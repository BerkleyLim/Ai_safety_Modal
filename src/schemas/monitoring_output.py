# src/schemas/monitoring_output.py
from typing import List, Optional
from pydantic import BaseModel, Field

class DetectedObject(BaseModel):
    """YOLO가 탐지한 단일 객체 정보"""
    class_name: str = Field(..., alias='class', description="탐지된 객체/이벤트 클래스 이름") 
    confidence: float = Field(..., ge=0.0, le=1.0, description="탐지 신뢰도 (0~1)")
    box: List[float] = Field(..., min_length=4, max_length=4, description="바운딩 박스 좌표 [x_min, y_min, x_max, y_max]")

    # [추가] 이 설정이 있어야 'class_name'으로 값을 넣을 수 있습니다.
    class Config:
        populate_by_name = True 

class MonitoringOutput(BaseModel):
    """Monitoring Layer의 최종 출력 스키마"""
    status: str = Field(..., description="'anomaly_detected' 또는 'normal'")
    image_path: str = Field(..., description="분석된 이미지 파일 경로")
    detected_objects: List[DetectedObject] = Field(default_factory=list, description="탐지된 객체 목록")

    # Pydantic 설정 (JSON 키 이름 'class'를 그대로 사용하기 위해)
    class Config:
        populate_by_name = True # alias 사용 활성화