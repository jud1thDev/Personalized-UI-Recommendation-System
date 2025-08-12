"""
Personalized UI Recommendation System - Colab Version
코랩에서 실행할 수 있는 통합 실행 파일 (모듈 기반)
"""

import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import yaml
import json
from datetime import datetime, timedelta
import random
from tqdm import tqdm
import math
from collections import Counter
from typing import List, Dict, Any, Literal

# 프로젝트 모듈 import
sys.path.append('ui_rec/src')
from models.ui_grouping import create_ui_component_groups
from utils.icons import get_icon_suggestion
from utils.io import read_yaml, ensure_dir, write_csv_with_timestamp, latest_file, save_model
from utils.features import entropy
from models.common import train_binary, train_multiclass, train_regression

# 코랩 환경 설정
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/models', exist_ok=True)
os.makedirs('data/outputs', exist_ok=True)
os.makedirs('config', exist_ok=True)

print("Personalized UI Recommendation System - Colab Version")
print("=" * 60)

# 1. 설정 파일 생성
print("1. 설정 파일 생성 중...")

data_config = {
    "paths": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed", 
        "models_dir": "data/models",
        "logs_dir": "data/logs",
        "outputs_dir": "data/outputs"
    },
    "files": {
        "events_pattern": "events_*.csv",
        "features_pattern": "features_*.csv"
    }
}

model_config = {
    "lgbm": {
        "exposure": {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "num_boost_round": 500,
            "early_stopping_rounds": 50
        },
        "ui_type": {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "num_boost_round": 500,
            "early_stopping_rounds": 50
        },
        "service_cluster": {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "num_boost_round": 500,
            "early_stopping_rounds": 50
        },
        "rank": {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "num_boost_round": 500,
            "early_stopping_rounds": 50
        }
    },
    "ui": {
        "allowed_component_types": ["card", "list_item", "banner", "icon", "grid_item"]
    },
    "service_clusters": {
        "mapping": {
            "f001": "account", "f002": "account", "f003": "account",
            "f004": "finance", "f005": "finance", "f006": "finance",
            "f007": "lifestyle", "f008": "lifestyle", "f009": "lifestyle",
            "f010": "health", "f011": "health", "f012": "health",
            "f013": "shopping", "f014": "shopping", "f015": "shopping",
            "f016": "travel", "f017": "travel", "f018": "travel",
            "f019": "recommendation", "f020": "recommendation"
        },
        "labels": {
            "account": "계좌/자산", "finance": "금융 서비스", "lifestyle": "생활 서비스",
            "health": "건강/포인트", "shopping": "쇼핑/혜택", "travel": "여행/교통", 
            "recommendation": "추천서비스"
        }
    }
}

# 설정 파일 저장
with open('config/data.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

with open('config/model.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(model_config, f, default_flow_style=False, allow_unicode=True)

print("설정 파일 생성 완료")

# 2. 모의 데이터 생성
print("2. 모의 데이터 생성 중...")

def access_time_cluster(ts: datetime) -> Literal["dawn","morning","afternoon","evening","night"]:
    """시간대별 클러스터 분류"""
    h = ts.hour
    if 5 <= h < 9: return "morning"
    if 9 <= h < 13: return "afternoon"
    if 13 <= h < 18: return "evening"
    if 18 <= h < 23: return "night"
    return "dawn"

# generate_mock.py에서 생성된 데이터 읽기
import glob

# 가장 최근 events 파일 찾기
events_files = glob.glob("data/raw/events_*.csv")
if not events_files:
    print("   events 파일을 찾을 수 없습니다. generate_mock.py를 먼저 실행해주세요.")
    sys.exit(1)

latest_events_file = max(events_files, key=os.path.getctime)
print(f"   데이터 파일 읽기: {latest_events_file}")

events_df = pd.read_csv(latest_events_file)
print(f"   총 {len(events_df)}개 이벤트 로드 완료")

# 사용자 정보 추출
users_df = events_df[["user_id","age_group","is_senior","device_type"]].drop_duplicates().reset_index(drop=True)
print(f"   총 {len(users_df)}명 사용자 정보 추출 완료")

# 기능 정보 추출
funcs_df = events_df[["function_id","service_cluster","title","subtitle"]].drop_duplicates().reset_index(drop=True)
print(f"   총 {len(funcs_df)}개 기능 정보 추출 완료")

# 3. 피처 생성 (build_features.py 모듈 직접 활용)
print("3. 피처 생성 중...")

# build_features.py 모듈 직접 실행
import subprocess
import sys

# 모듈 실행
result = subprocess.run([sys.executable, "-m", "ui_rec.src.features.build_features"], 
                       capture_output=True, text=True, cwd=".")
if result.returncode == 0:
    print("   피처 생성 완료")
    # 생성된 피처 파일 읽기
    from ui_rec.src.utils.io import latest_file
    features_path = latest_file("features_*.csv", "data/processed")
    features_df = pd.read_csv(features_path)
    print(f"   총 {len(features_df)}개 샘플, {len(features_df.columns)}개 피처")
else:
    print(f"   피처 생성 실패: {result.stderr}")
    sys.exit(1)

# 4. 모델 학습 (common.py 모듈 활용)
print("4. 모델 학습 중...")

# 각 모델 학습
exposure_model_path = train_binary(features_df, "exposure_label", "lgbm_exposure")
ui_type_model_path = train_multiclass(features_df, "ui_type_label", "ui_type", "lgbm_ui_type")
service_cluster_model_path = train_multiclass(features_df, "service_cluster_label", "service_cluster", "lgbm_service_cluster")
rank_model_path = train_regression(features_df, "rank_label", "lgbm_rank")

print("모든 모델 학습 완료")

# 5. 추론 및 결과 생성
print("5. 추론 및 결과 생성 중...")

def predict_and_generate_output_colab(features_df, allowed_types):
    """추론 및 JSON 출력 생성 (코랩용)"""
    print("   추론 실행 중...")
    
    # 입력 데이터 준비
    drop_cols = ["exposure_label","ui_type_label","service_cluster_label","rank_label"]
    X = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns])
    
    # 피처 전처리
    X_processed, _ = preprocess_features(features_df, "exposure_label", drop_cols)
    
    # 예측
    exp_prob = exposure_model.predict(X_processed)
    exp_pred = (exp_prob > 0.7).astype(int)
    
    ui_raw = ui_type_model.predict(X_processed)
    ui_idx = np.argmax(ui_raw, axis=1)
    ui_label = [
        model_config["ui"]["allowed_component_types"][i % len(model_config["ui"]["allowed_component_types"])]
        for i in ui_idx
    ]
    
    grp_raw = service_cluster_model.predict(X_processed)
    grp_idx = np.argmax(grp_raw, axis=1)
    grp_label = [list(model_config["service_clusters"]["labels"].keys())[i % len(model_config["service_clusters"]["labels"])] for i in grp_idx]
    
    rank_pred = rank_model.predict(X_processed)
    
    # 사용자별 JSON 생성
    outputs = []
    for user_id, udf in features_df.assign(include=exp_pred, ui=ui_label, grp=grp_label, order=rank_pred).groupby("user_id"):
        # 레이아웃 밀도 결정
        is_senior = bool(udf.is_senior.iloc[0])
        path_entropy = float(udf.tap_path_entropy.iloc[0])
        
        if is_senior:
            layout = "low" if path_entropy < 1.2 else "medium"
        else:
            layout = "medium" if path_entropy < 1.5 else "high"
        
        # 노출할 기능들만 필터링
        exposed_functions = []
        for _, row in udf.iterrows():
            if int(row["include"]) == 0:
                continue
            
            # 허용된 UI 타입 사용 (아이콘 포함)
            ui_type = str(row["ui"])
            if ui_type not in allowed_types:
                ui_type = "card"  # 허용되지 않은 타입만 카드로 대체
            
            # 서비스 클러스터에 따른 추천 아이콘 결정
            icon_suggestion = get_icon_suggestion(str(row["grp"]))
            
            exposed_functions.append({
                "function_id": row["function_id"],
                "component_type": ui_type,
                "service_cluster": str(row["grp"]),
                "order": float(row["order"]),
                "icon_suggestion": icon_suggestion  # 아이콘 제안 정보 포함
            })
        
        # ui_grouping.py를 사용해서 그룹화
        groups = create_ui_component_groups(exposed_functions, user_name=f"U{user_id}")
        
        # 최종 출력 구조
        output = {
            "user_id": user_id,
            "home": {
                "layout_density": layout,
                "groups": groups
            }
        }
        outputs.append(output)
    
    # JSON 저장
    output_path = "data/outputs/ui_home_outputs.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    
    print(f"추론 완료: {output_path}")
    print(f"   총 {len(outputs)}명 사용자에 대한 홈 화면 구성 생성")
    
    return outputs

def preprocess_features(df, target_col, drop_cols=None):
    """피처 전처리 - 모든 피처를 숫자로 변환"""
    if drop_cols is None:
        drop_cols = []
    
    X = df.drop(columns=[target_col] + drop_cols)
    
    # 모든 object 컬럼을 범주형 코드로 변환
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category").cat.codes
    
    # bool은 정수로 변환
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype("int8")
    
    # category 컬럼을 정수로 변환
    for col in X.select_dtypes(include=["category"]).columns:
        X[col] = X[col].cat.codes
    
    y = df[target_col]
    
    # 타겟 변수가 문자열인 경우 숫자로 인코딩
    if y.dtype == "object":
        y = y.astype("category").cat.codes
    
    return X, y

# 모델 로드
    import joblib
exposure_model = joblib.load(exposure_model_path)
ui_type_model = joblib.load(ui_type_model_path)
service_cluster_model = joblib.load(service_cluster_model_path)
rank_model = joblib.load(rank_model_path)

# 추론 실행
allowed_types = ["card", "list_item", "banner", "icon"]
results = predict_and_generate_output_colab(features_df, allowed_types)

# 6. 결과 요약
print("6. 실행 완료!")
print("=" * 60)
print(f"데이터: {len(events_df)}개 이벤트, {len(users_df)}명 사용자")
print(f"피처: {len(features_df)}개 샘플, {len(features_df.columns)}개 피처")
print(f"모델: 4개 모델 학습 완료")
print(f"결과: {len(results)}명 사용자 홈 화면 구성 생성")
print(f"출력: data/outputs/ui_home_outputs.json")
print("모든 작업이 성공적으로 완료되었습니다!")

# 결과 미리보기
print("결과 미리보기:")
if results:
    sample_user = results[0]
    print(f"사용자: {sample_user['user_id']}, 레이아웃: {sample_user['home']['layout_density']}, 그룹: {len(sample_user['home']['groups'])}개")
    if sample_user['home']['groups']:
        sample_group = sample_user['home']['groups'][0]
        print(f"첫 번째 그룹: {sample_group['label']} ({len(sample_group['functions'])}개 기능)")
