import os
import glob
import random
import argparse
import sys
import json
import time  # [추가] 시간 측정용
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dotenv import load_dotenv

# =========================================================
# [설정] 프로젝트 경로 및 모듈 임포트
# =========================================================
FILE_PATH = Path(__file__).resolve()
SRC_DIR = FILE_PATH.parent.parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.append(str(SRC_DIR))
sys.path.append(str(PROJECT_ROOT))

try:
    from monitoring import detect_objects
    from reasoning import analyze_risk_with_vlm
    from schemas.monitoring_output import MonitoringOutput
except ImportError as e:
    print(f"🚨 모듈 임포트 오류: {e}")
    sys.exit(1)

load_dotenv()

# =========================================================
# [설정] 정답지 기준 (Ground Truth)
# =========================================================
ID_TO_CLASS = {
    # --- Static Object (SO) ---
    0: "SO-01", 1: "SO-02", 2: "SO-03", 3: "SO-05", 4: "SO-06", 
    5: "SO-07", 6: "SO-08", 7: "SO-09", 8: "SO-10", 9: "SO-11", 
    10: "SO-12", 11: "SO-13", 12: "SO-14", 13: "SO-15", 14: "SO-16", 
    15: "SO-17", 16: "SO-18", 17: "SO-19", 18: "SO-21", 19: "SO-22", 
    20: "SO-23",

    # --- Work Object (WO) ---
    21: "WO-01", 22: "WO-02", 23: "WO-03", 24: "WO-04", 
    25: "WO-05", 26: "WO-06", 27: "WO-07", 28: "WO-08",

    # --- Unsafe Action (UA) ---
    29: "UA-01", 30: "UA-02", 31: "UA-03", 32: "UA-04", 33: "UA-05", 
    34: "UA-06", 35: "UA-10", 36: "UA-12", 37: "UA-13", 38: "UA-14", 
    39: "UA-16", 40: "UA-17", 41: "UA-20",

    # --- Unsafe Condition (UC) ---
    42: "UC-02", 43: "UC-06", 44: "UC-08", 45: "UC-09", 46: "UC-10", 
    47: "UC-13", 48: "UC-14", 49: "UC-15", 50: "UC-16", 51: "UC-17", 
    52: "UC-18", 53: "UC-19", 54: "UC-20", 55: "UC-21", 56: "UC-22"
}

RISK_CLASSES = [
    "UA-01", "UA-02", "UA-03", "UA-04", "UA-05", "UA-06", "UA-10",
    "UA-12", "UA-13", "UA-14", "UA-16", "UA-17", "UA-20",
    "UC-02", "UC-06", "UC-08", "UC-09", "UC-10", "UC-13", "UC-14",
    "UC-15", "UC-16", "UC-17", "UC-18", "UC-19", "UC-20", "UC-21", "UC-22",
    "SO-21", "SO-22"
]

def get_ground_truth_code(image_path):
    label_path = str(image_path).replace("/images/", "/labels/").replace(".jpg", ".txt").replace(".png", ".txt")
    if not os.path.exists(label_path):
        return "NONE"
    detected_risks = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                try:
                    cls_id = int(parts[0])
                    cls_name = ID_TO_CLASS.get(cls_id)
                    if cls_name and cls_name in RISK_CLASSES:
                        detected_risks.append(cls_name)
                except ValueError: continue
    except Exception: pass
    return detected_risks[0] if detected_risks else "NONE"

class VLMEvaluator:
    def __init__(self, data_root: str, sample_size: int = 50):
        self.data_root = Path(data_root)
        self.sample_size = sample_size
        self.val_images_dir = self.data_root / "val" / "images"

    def run(self, output_csv="vlm_evaluation_result.csv", mode="hybrid"):
        if not self.val_images_dir.exists():
            print(f"🚨 오류: 검증 데이터 경로 없음 ({self.val_images_dir})")
            return

        image_files = sorted(list(self.val_images_dir.glob("*.jpg")) + list(self.val_images_dir.glob("*.png")))
        print(f"📂 전체 검증 이미지: {len(image_files)}장")

        if self.sample_size and len(image_files) > self.sample_size:
            print(f"🎲 {self.sample_size}장 랜덤 샘플링 (Seed=42)...")
            random.seed(42)
            image_files = random.sample(image_files, self.sample_size)
        
        results = []
        y_true_binary = []
        y_pred_binary = []
        y_true_code = []
        y_pred_code = []
        
        total_yolo_time = 0.0
        total_vlm_time = 0.0
        total_proc_time = 0.0
        vlm_call_count = 0

        print(f"🚀 평가 시작 (Mode: {mode.upper()})...")
        
        for i, img_path in enumerate(tqdm(image_files)):
            t_start = time.time()
            t_yolo = 0.0
            t_vlm = 0.0
            
            # 1. 정답(GT)
            gt_code = get_ground_truth_code(str(img_path))
            gt_binary = "ANOMALY" if gt_code != "NONE" else "NORMAL"
            
            # 2. 파이프라인 실행
            detection_result = None
            reasoning_result = None
            
            # (A) Monitoring (YOLO)
            if mode in ["hybrid", "vlm-evaluate"]:
                t0 = time.time()
                detection_result = detect_objects(str(img_path))
                t1 = time.time()
                t_yolo = t1 - t0
            else:
                # vlm-only
                detection_result = MonitoringOutput(status="normal", image_path=str(img_path), detected_objects=[])
                t_yolo = 0.0

            # (B) Reasoning (VLM)
            should_run_vlm = False
            if mode in ["vlm-evaluate", "vlm-only"]:
                should_run_vlm = True
            elif mode == "hybrid":
                if detection_result.status == "anomaly_detected":
                    should_run_vlm = True
            
            if should_run_vlm:
                t2 = time.time()
                reasoning_result = analyze_risk_with_vlm(detection_result)
                t3 = time.time()
                t_vlm = t3 - t2
                vlm_call_count += 1
            
            t_end = time.time()
            t_total = t_end - t_start
            
            total_yolo_time += t_yolo
            total_vlm_time += t_vlm
            total_proc_time += t_total

            # 3. 예측 결과
            pred_code = "NONE"
            pred_binary = "NORMAL"
            reason = "Skipped (Safe)"
            
            if reasoning_result:
                pred_code = reasoning_result.hazard_code if reasoning_result.hazard_code else "NONE"
                reason = reasoning_result.reason
                
                clean_code = str(pred_code).upper().strip()
                SAFE_KEYWORDS = ["NONE", "SAFE", "N/A", "NULL", "NONE"]
                
                if (clean_code not in SAFE_KEYWORDS) or (reasoning_result.risk_level in ["HIGH", "MED", "LOW"]):
                     pred_binary = "ANOMALY"
                else:
                     pred_binary = "NORMAL"
            elif mode == "hybrid" and not should_run_vlm:
                # YOLO가 Normal -> VLM 스킵 -> Normal 예측
                pred_binary = "NORMAL"
                pred_code = "NONE"

            # 4. 채점 및 데이터 수집
            is_binary_correct = (gt_binary == pred_binary)
            is_code_correct = (gt_code == pred_code)

            results.append({
                "Image": img_path.name,
                "GT_Code": gt_code,
                "Pred_Code": pred_code,
                "GT_Binary": gt_binary,
                "Pred_Binary": pred_binary,
                "Acc_Binary": is_binary_correct,
                "Acc_Code": is_code_correct,
                "Time_Total": t_total,
                "Reason":reason
            })
            
            y_true_binary.append(gt_binary)
            y_pred_binary.append(pred_binary)
            
            # [추가] 실제 위험 상황에 대해서만 코드 식별 정확도 계산을 위해 데이터 수집
            if gt_binary == "ANOMALY":
                y_true_code.append(gt_code)
                y_pred_code.append(pred_code)

        # 5. 결과 리포트
        if not results:
            print("❌ 결과 없음")
            return

        df = pd.DataFrame(results)
        output_path = Path(output_csv)
        if output_path.parent:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 결과 저장: {output_csv}")
        print(f"\n📊 [평가 결과 - Mode: {mode.upper()}] (N={len(results)})")
        
        print(f"1. 🛡️ 위험 감지 정확도 (Binary): {df['Acc_Binary'].mean():.2%}")
        print(f"2. 🎯 위험 식별 정확도 (Code):   {df['Acc_Code'].mean():.2%}")
        
        if y_true_code:
            risk_code_acc = accuracy_score(y_true_code, y_pred_code)
            print(f"   - 실제 위험 데이터 식별 정확도: {risk_code_acc:.2%}")
        else:
            print("   - 실제 위험 데이터가 없어 코드 식별 정확도를 계산할 수 없습니다.")

        avg_yolo = total_yolo_time / len(results)
        avg_vlm = total_vlm_time / len(results)
        avg_total = total_proc_time / len(results)
        vlm_rate = vlm_call_count / len(results)
        
        print(f"\n⏱️ [효율성 분석]")
        print(f"   - 평균 YOLO 시간: {avg_yolo:.4f} sec")
        print(f"   - 평균 VLM  시간: {avg_vlm:.4f} sec")
        print(f"   - 평균 전체 시간: {avg_total:.4f} sec")
        print(f"   - VLM 호출 비율:  {vlm_rate:.2%} ({vlm_call_count}/{len(results)})")
        
        print("\n📑 [혼동 행렬]")
        print(confusion_matrix(y_true_binary, y_pred_binary, labels=["ANOMALY", "NORMAL"]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--sample', type=int, default=50)
    parser.add_argument('--output', type=str, default='logs/eval_result.csv')
    parser.add_argument('--mode', type=str, choices=['vlm-evaluate', 'vlm-only', 'hybrid'], default='hybrid')
    args = parser.parse_args()
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("🚨 API Key 필요")
        return
    
    evaluator = VLMEvaluator(args.data_root, args.sample)
    evaluator.run(args.output, args.mode)

if __name__ == "__main__":
    main()