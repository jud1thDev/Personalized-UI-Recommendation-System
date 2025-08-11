import argparse
import json
import os
import pandas as pd
import numpy as np

from ..utils.io import read_yaml, latest_file, load_model, ensure_dir
from ..utils.schema import ui_response_template
from ..models.common import apply_feature_mapping

CFG_DATA = read_yaml("ui_rec/config/data.yaml")
CFG_MODEL = read_yaml("ui_rec/config/model.yaml")
PROC_DIR = CFG_DATA["paths"]["processed_dir"]
MODELS_DIR = CFG_DATA["paths"]["models_dir"]
OUTPUTS_DIR = CFG_DATA["paths"]["outputs_dir"]


def preprocess_features(df: pd.DataFrame, feature_mapping: dict) -> pd.DataFrame:
    """범주형 피처 문제를 해결하기 위해 모든 피처를 숫자로 변환"""
    X = df.copy()
    
    # 모든 object 컬럼을 범주형 코드로 변환
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category").cat.codes
    
    # bool 컬럼을 정수로 변환
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype("int8")
    
    # category 컬럼을 정수로 변환
    for col in X.select_dtypes(include=["category"]).columns:
        X[col] = X[col].cat.codes
    
    return X


def predict_allowed_ui(pred_label: str, allowed: list[str]) -> str:
    return pred_label if pred_label in allowed else (allowed[0] if allowed else pred_label)


def layout_density_rule(is_senior: bool, path_entropy: float) -> str:
    if is_senior:
        return "low" if path_entropy < 1.2 else "medium"
    return "medium" if path_entropy < 1.5 else "high"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowed", type=str, default="")
    args = parser.parse_args()
    allowed = [t for t in (args.allowed.split(",") if args.allowed else []) if t]
    if not allowed:
        allowed = CFG_MODEL["ui"]["allowed_component_types"]

    feat_path = latest_file(CFG_DATA["files"]["features_pattern"], PROC_DIR)
    if not feat_path:
        raise FileNotFoundError("No features CSV. Build features first.")
    df = pd.read_csv(feat_path)

    # 로드 모델들
    exposure = load_model(os.path.join(MODELS_DIR, "lgbm_exposure.joblib"))
    ui_type = load_model(os.path.join(MODELS_DIR, "lgbm_ui_type.joblib"))
    group = load_model(os.path.join(MODELS_DIR, "lgbm_group_label.joblib"))
    rank = load_model(os.path.join(MODELS_DIR, "lgbm_rank.joblib"))

    # 입력 X 구성(필요 타깃/누설 변수 제거)
    drop_cols = ["exposure_label","ui_type_label","service_cluster_label","label_text","rank_label"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # 학습 시와 동일한 피처 컬럼 순서 보장 (저장된 매핑 사용)
    feature_mapping = exposure['feature_mapping']
    expected_features = feature_mapping['column_order']
    
    # 필요한 컬럼만 선택하고 순서 맞춤
    available_features = [col for col in expected_features if col in X.columns]
    X = X[available_features]
    
    # 학습과 동일한 피처 전처리 적용 (exposure 모델의 피처 매핑 사용)
    X = preprocess_features(X, exposure['feature_mapping'])

    # 예측
    exp_prob = exposure['model'].predict(X, num_iteration=exposure['model'].best_iteration)
    exp_pred = (exp_prob > 0.5).astype(int)

    ui_raw = ui_type['model'].predict(X, num_iteration=ui_type['model'].best_iteration)
    ui_idx = np.argmax(ui_raw, axis=1)
    ui_label = [ui_type['classes'][i] for i in ui_idx]

    grp_raw = group['model'].predict(X, num_iteration=group['model'].best_iteration)
    grp_idx = np.argmax(grp_raw, axis=1)
    grp_label = [group['classes'][i] for i in grp_idx]

    rank_pred = rank['model'].predict(X, num_iteration=rank['model'].best_iteration)

    # 사용자별 JSON 생성
    outputs = []
    for user_id, udf in df.assign(include=exp_pred, ui=ui_label, grp=grp_label, order=rank_pred).groupby("user_id"):
        layout = layout_density_rule(bool(udf.is_senior.iloc[0]), float(udf.tap_path_entropy.iloc[0]))
        items = []
        for _, row in udf.iterrows():
            if int(row["include"]) == 0:
                continue
            items.append({
                "function_id": row["function_id"],
                "include": True,
                "component_type": predict_allowed_ui(str(row["ui"]), allowed),
                "service_cluster": str(row["grp"]),
                "label": CFG_MODEL["service_clusters"]["labels"].get(str(row["grp"]), str(row.get("label_text",""))),
                "order": float(row["order"])  # 프론트에서 정렬
            })
        items = sorted(items, key=lambda x: x["order"])  # 오름차순 배치
        outputs.append(ui_response_template(user_id, layout, items))

    ensure_dir(OUTPUTS_DIR)
    out_path = os.path.join(OUTPUTS_DIR, "ui_home_outputs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main() 