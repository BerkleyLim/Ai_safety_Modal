# src/schemas/reasoning_output.py
from typing import Optional
from pydantic import BaseModel, Field

class ReasoningOutput(BaseModel):
    """Reasoning Layer의 최종 출력 (VLM 분석 결과) 스키마"""
    risk_level: str = Field(..., description="'LOW', 'MED', 'HIGH' 중 하나")
    
    # VLM 프롬프트의 새 출력 필드 추가
    hazard_code: Optional[str] = Field("NONE", description="식별된 위험 Class ID (없으면 NONE)")
    
    reason: str = Field(..., description="VLM이 분석한 위험 발생 이유 (한국어)")
    image_path: Optional[str] = Field(None, description="분석된 이미지 파일 경로 (Action Layer 전달용)")