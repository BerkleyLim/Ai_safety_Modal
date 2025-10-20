# src/action.py

def generate_safety_guideline(analysis_result):
    """
    [가짜 Action Layer 함수]
    실제로는 여기서 LLM API를 호출하여 다국어 안전 지침을 생성합니다.
    지금은 분석 결과에 따라 미리 정해진 문구를 출력합니다.
    """
    print("📢 [Action] 분석 결과 기반 안전 지침 생성 중...")
    
    risk_level = analysis_result.get("risk_level", "Unknown")
    reason = analysis_result.get("reason", "No reason provided.")
    
    if risk_level == "High":
        guideline_ko = f"🚨 긴급 경고! 심각한 위험이 감지되었습니다. 원인: {reason}. 즉시 작업을 중단하고 안전 관리자에게 보고하십시오!"
        guideline_en = f"🚨 URGENT WARNING! High risk detected. Reason: {reason}. Stop work immediately and report to the safety manager!"
        
        print("--- [생성된 안전 지침] ---")
        print(f"🇰🇷 (한국어): {guideline_ko}")
        print(f"🇺🇸 (English): {guideline_en}")
        print("--------------------------")
    else:
        print("안전 상태 양호. 상황을 계속 주시합니다.")

    return {"status": "guideline_generated"}