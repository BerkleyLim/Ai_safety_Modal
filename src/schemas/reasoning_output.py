# src/schemas/reasoning_output.py
from typing import Optional
from pydantic import BaseModel, Field

class ReasoningOutput(BaseModel):
    """Reasoning Layer의 최종 출력 (VLM 분석 결과) 스키마"""
    risk_level: str = Field(..., description="'LOW', 'MED', 'HIGH' 중 하나")
    reason: str = Field(..., description="VLM이 분석한 위험 발생 이유 (한국어)")
    image_path: Optional[str] = Field(None, description="분석된 이미지 파일 경로 (선택적)")