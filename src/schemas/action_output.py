# src/schemas/action_output.py
from typing import Optional, Dict
from pydantic import BaseModel, Field

class MultilingualGuidelines(BaseModel):
    """LLM이 생성한 다국어 안전 지침"""
    guideline_ko: Optional[str] = Field(None, description="한국어 안전 지침")
    guideline_en: Optional[str] = Field(None, description="영어 안전 지침")
    guideline_vi: Optional[str] = Field(None, description="베트남어 안전 지침")
    # 필요한 다른 언어 추가 가능

class ActionOutput(BaseModel):
    """Action Layer의 최종 출력 스키마"""
    status: str = Field(..., description="수행된 조치 상태 ('logged', 'confirmation_requested', 'multilingual_guideline_generated', 'error_...')")
    risk_level_processed: Optional[str] = Field(None, description="처리된 위험 등급 ('LOW', 'MED', 'HIGH')")
    reason_detected: Optional[str] = Field(None, description="Reasoning Layer에서 전달된 위험 발생 이유")
    guidelines: Optional[MultilingualGuidelines] = Field(None, description="HIGH 위험 시 생성된 다국어 안전 지침")