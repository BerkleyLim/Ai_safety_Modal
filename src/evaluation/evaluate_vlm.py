import os
import glob
import random
import argparse
import sys
import json
import time
import csv
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
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
# [설정] 정답지 기준
# =========================================================
RISK_CLASSES = [
    "UA-01", "UA-02", "UA-03", "UA-04", "UA-05", "UA-06", "UA-10",
    "UA-12", "UA-13", "UA-14", "UA-16", "UA-17", "UA-20",
    "UC-02", "UC-06", "UC-08", "UC-09", "UC-10", "UC-13", "UC-14",
    "UC-15", "UC-16", "UC-17", "UC-18", "UC-19", "UC-20", "UC-21", "UC-22",
    "SO-21", "SO-22"
]

def load_mapping(mapping_csv_path):
    """filename_mapping.csv를 읽어서 {새파일명: 원본경로} 딕셔너리 반환"""
    mapping = {}
    if not os.path.exists(mapping_csv_path):
        print(f"⚠️ 매핑 파일을 찾을 수 없습니다: {mapping_csv_path}")
        return mapping
        
    try:
        with open(mapping_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row['New_Filename']] = row['Original_Path']
        print(f"✅ 매핑 파일 로드 완료: {len(mapping)}개 파일 정보")
    except Exception as e:
        print(f"🚨 매핑 파일 읽기 오류: {e}")
        
    return mapping

def get_ground_truth_from_json(image_name, mapping_data):
    """
    매핑 정보를 이용해 원본 JSON을 찾아 진짜 정답(Ground Truth)을 반환
    """
    original_img_path = mapping_data.get(image_name)
    if not original_img_path:
        return []

    # 1. 원본 이미지 경로 -> 원본 JSON 경로 변환
    # 규칙: /original/ -> /label/
    # 규칙: TS_ -> TL_, VS_ -> VL_ (폴더명 접두사 변경)
    # 규칙: .jpg/.png -> .json
    
    json_path = original_img_path.replace("/original/", "/label/")
    json_path = json_path.replace("TS_", "TL_").replace("VS_", "VL_")
    json_path = os.path.splitext(json_path)[0] + ".json"
    
    if not os.path.exists(json_path):
        return []

    detected_risks = set()
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # (A) Situation ID 확인 (가장 중요한 정답)
            raw_info = data.get("Raw data Info.", {})
            sit_id = raw_info.get("situation_ID")
            if sit_id in RISK_CLASSES:
                detected_risks.add(sit_id)
                
            # (B) Annotation 확인 (보조 정답)
            annotations = data.get("Learning data info.", {}).get("annotation", [])
            for ann in annotations:
                cls_id = ann.get("class_id")
                if cls_id in RISK_CLASSES:
                    detected_risks.add(cls_id)
                    
    except Exception as e:
        print(f"⚠️ JSON 파싱 오류 ({json_path}): {e}")
        return []

    return list(detected_risks)


class VLMEvaluator:
    def __init__(self, data_root: str, mapping_path: str, sample_size: int = 50):
        self.data_root = Path(data_root)
        self.sample_size = sample_size
        self.val_images_dir = self.data_root / "val" / "images"
        
        # [수정] 여기서 매핑 데이터를 로드해서 self.mapping_data에 저장해야 합니다!
        self.mapping_data = load_mapping(mapping_path)

    def run(self, output_csv="vlm_evaluation_result.csv", mode="hybrid"):
        if not self.val_images_dir.exists():
            print(f"🚨 오류: 검증 데이터 경로 없음 ({self.val_images_dir})")
            return

        image_files = sorted(list(self.val_images_dir.glob("*.jpg")) + list(self.val_images_dir.glob("*.png")))
        print(f"📂 전체 검증 이미지: {len(image_files)}장")

        if self.sample_size and len(image_files) > self.sample_size:
            print(f"🎲 {self.sample_size}장 랜덤 샘플링...")
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
        
        for img_path in tqdm(image_files):
            t_start = time.time()
            t_yolo = 0.0
            t_vlm = 0.0
            
            # 1. 정답(GT) 확인 - 원본 JSON 기반
            gt_codes = get_ground_truth_from_json(img_path.name, self.mapping_data)
            gt_binary = "ANOMALY" if gt_codes else "NORMAL"
            gt_codes_str = ", ".join(gt_codes) if gt_codes else "NONE"
            
            # 2. 파이프라인 실행
            detection_result = None
            reasoning_result = None
            
            # (A) YOLO
            if mode in ["hybrid", "vlm-evaluate"]:
                t0 = time.time()
                detection_result = detect_objects(str(img_path))
                t_yolo = time.time() - t0
            else:
                detection_result = MonitoringOutput(status="normal", image_path=str(img_path), detected_objects=[])
                t_yolo = 0.0

            # (B) VLM
            should_run_vlm = False
            if mode in ["vlm-evaluate", "vlm-only"]:
                should_run_vlm = True
            elif mode == "hybrid" and detection_result.status == "anomaly_detected":
                should_run_vlm = True
            
            if should_run_vlm:
                t2 = time.time()
                reasoning_result = analyze_risk_with_vlm(detection_result)
                t_vlm = time.time() - t2
                vlm_call_count += 1
            
            t_total = time.time() - t_start
            total_yolo_time += t_yolo
            total_vlm_time += t_vlm
            total_proc_time += t_total

            # 3. 예측 결과 파싱
            pred_code = "NONE"
            pred_binary = "NORMAL"
            reason = "Skipped"
            
            if reasoning_result:
                pred_code = reasoning_result.hazard_code if reasoning_result.hazard_code else "NONE"
                reason = reasoning_result.reason
                
                clean_code = str(pred_code).upper().strip()
                SAFE_KEYWORDS = ["NONE", "SAFE", "N/A", "NULL", "NONE"]
                
                if clean_code not in SAFE_KEYWORDS:
                     pred_binary = "ANOMALY"
                else:
                     pred_binary = "NORMAL"
            elif mode == "hybrid" and not should_run_vlm:
                pred_binary = "NORMAL"
                pred_code = "NONE"
                reason = "Skipped by YOLO"

            # 4. 채점
            is_binary_correct = (gt_binary == pred_binary)
            
            # 복수 정답 인정 로직
            if pred_code in gt_codes:
                is_code_correct = True
            elif pred_code == "NONE" and not gt_codes:
                is_code_correct = True
            else:
                is_code_correct = False

            results.append({
                "Image": img_path.name,
                "GT_Codes": gt_codes_str,
                "Pred_Code": pred_code,
                "GT_Binary": gt_binary,
                "Pred_Binary": pred_binary,
                "Acc_Binary": is_binary_correct,
                "Acc_Code": is_code_correct,
                "Reason": reason,
                "Time_Total": t_total
            })
            
            y_true_binary.append(gt_binary)
            y_pred_binary.append(pred_binary)
            
            if gt_binary == "ANOMALY":
                # 정확도 통계용: 맞췄으면 정답 코드 사용, 틀렸으면 첫 번째 GT 사용
                target_gt = pred_code if is_code_correct else gt_codes[0]
                y_true_code.append(target_gt)
                y_pred_code.append(pred_code)

        # 5. 결과 저장
        if not results:
            return

        df = pd.DataFrame(results)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        print(f"\n📊 [평가 결과 - Mode: {mode.upper()}] (N={len(results)})")
        print(f"1. 🛡️ 위험 감지 정확도 (Binary): {df['Acc_Binary'].mean():.2%}")
        print(f"2. 🎯 위험 식별 정확도 (Code):   {df['Acc_Code'].mean():.2%}")
        
        if y_true_code:
            risk_code_acc = accuracy_score(y_true_code, y_pred_code)
            print(f"   - (위험 데이터 대상) 식별 정확도: {risk_code_acc:.2%}")

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
    parser.add_argument('--mapping-csv', type=str, required=True, help='매핑 파일 경로')
    parser.add_argument('--sample', type=int, default=50)
    parser.add_argument('--output', type=str, default='logs/eval_result.csv')
    parser.add_argument('--mode', type=str, default='hybrid')
    args = parser.parse_args()
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("🚨 API Key 필요")
        return
        
    # [수정] mapping_csv 인자 전달
    evaluator = VLMEvaluator(args.data_root, args.mapping_csv, args.sample)
    evaluator.run(args.output, args.mode)

if __name__ == "__main__":
    main()