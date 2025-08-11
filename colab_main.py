"""
Personalized UI Recommendation System - Colab Version
코랩에서 실행할 수 있는 통합 실행 파일
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
        "group_label": {
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
            "f001": "finance", "f002": "finance", "f003": "lifestyle", "f004": "lifestyle",
            "f005": "shopping", "f006": "shopping", "f007": "health", "f008": "health",
            "f009": "travel", "f010": "travel"
        },
        "labels": {
            "finance": "금융 추천", "lifestyle": "라이프스타일", "shopping": "쇼핑 추천",
            "health": "건강 관리", "travel": "여행/교통"
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

def generate_mock_data(n_users: int = 2000):
    """모의 사용자 행동 데이터 생성"""
    print(f"   {n_users}명 사용자 데이터 생성 중...")
    
    age_groups = ["10s","20s","30s","40s","50s","60s","70s+"]
    devices = ["Android","iOS"]
    
    # 사용자 생성
    users = []
    for i in range(n_users):
        uid = f"U{i:05d}"
        ag = np.random.choice(age_groups, p=[0.08,0.22,0.24,0.2,0.14,0.09,0.03])
        is_senior = ag in ["60s","70s+"]
        device = np.random.choice(devices, p=[0.65,0.35])
        users.append((uid, ag, is_senior, device))
    
    users_df = pd.DataFrame(users, columns=["user_id","age_group","is_senior","device_type"])
    
    # 기능 생성
    functions = []
    for i in range(20):
        fid = f"f{i+1:03d}"
        mapping = model_config["service_clusters"]["mapping"]
        cluster = mapping.get(fid, np.random.choice(list(model_config["service_clusters"]["labels"].keys())))
        label = model_config["service_clusters"]["labels"][cluster]
        functions.append((fid, cluster, label))
    
    funcs_df = pd.DataFrame(functions, columns=["function_id","service_cluster","label"])
    
    # 이벤트 시뮬레이션
    print("   사용자 행동 이벤트 시뮬레이션 중...")
    rows = []
    start = datetime.now() - timedelta(days=30)
    allowed_types = model_config["ui"]["allowed_component_types"]
    
    for _, u in tqdm(users_df.iterrows(), total=len(users_df)):
        user_base_sessions = np.random.poisson(lam=2.0) + 1 
        for d in range(30):
            day = start + timedelta(days=d)
            n_sessions = max(0, int(np.random.normal(user_base_sessions, 0.8)))
            for s in range(n_sessions):
                session_id = f"{u.user_id}-{day.strftime('%Y%m%d')}-{s:02d}"
                n_events = np.random.randint(2, 8)  
                current_funcs = funcs_df.sample(np.random.randint(3, min(8, len(funcs_df))))
                for _ in range(n_events):
                    f = current_funcs.sample(1).iloc[0]
                    ts = day + timedelta(minutes=int(np.random.uniform(0, 24*60)))
                    dwell = max(1, np.random.exponential(scale=60))
                    clicked = np.random.rand() < 0.35
                    component_type = np.random.choice(allowed_types)
                    component_id = f"cmp-{f.function_id}-{np.random.randint(1000):04d}"
                    position = np.random.randint(1, 50)
                    rows.append([
                        u.user_id, u.age_group, u.is_senior, u.device_type,
                        f.function_id, f.service_cluster, f.label,
                        session_id, ts, access_time_cluster(ts),
                        dwell, int(clicked), component_id, position, component_type
                    ])
    
    cols = [
        "user_id","age_group","is_senior","device_type",
        "function_id","service_cluster","label",
        "session_id","timestamp","access_time_cluster",
        "dwell_seconds","clicked","component_id","component_position","component_type"
    ]
    
    events_df = pd.DataFrame(rows, columns=cols)
    
    # CSV 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    events_path = f"data/raw/events_{timestamp}.csv"
    events_df.to_csv(events_path, index=False)
    
    print(f"모의 데이터 생성 완료: {events_path}")
    print(f"   총 {len(events_df)}개 이벤트, {len(users_df)}명 사용자")
    
    return events_df, users_df, funcs_df

events_df, users_df, funcs_df = generate_mock_data()

# 3. 피처 생성
print("3. 피처 생성 중...")

def entropy(sequence: List[str]) -> float:
    """시퀀스의 엔트로피 계산"""
    if not sequence: return 0.0
    cnt = Counter(sequence)
    total = sum(cnt.values())
    ent = 0.0
    for c in cnt.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent

def build_features(events_df):
    """사용자 행동 데이터로부터 피처 생성"""
    print("   사용자 레벨 피처 생성 중...")
    
    # 사용자 레벨 경로 엔트로피, 세션 수 등
    path_entropy = events_df.groupby(["user_id","session_id"])['function_id'].apply(list)
    user_entropy = path_entropy.groupby(level=0).apply(lambda s: np.mean([entropy(seq) for seq in s]))
    
    session_count = events_df.groupby(["user_id","session_id"]).size().groupby(level=0).size()
    fav_time = events_df.groupby("user_id")["access_time_cluster"].agg(lambda x: x.value_counts().index[0])
    
    user_df = pd.DataFrame({
        "tap_path_entropy": user_entropy,
        "session_count": session_count,
        "access_time_cluster": fav_time
    }).reset_index()
    
    print("   사용자-기능 단위 피처 생성 중...")
    
    # 사용자-기능 단위 집계
    grp = events_df.groupby(["user_id","function_id"])  
    entry_count = grp.size().rename("entry_count")
    click_rate = grp["clicked"].mean().rename("click_rate")
    visit_duration = grp["dwell_seconds"].mean().rename("visit_duration")
    
    # 재방문: 같은 기능 1시간 이내 재진입 횟수 근사
    events_sorted = events_df.sort_values(["user_id","function_id","timestamp"])  
    events_sorted["prev_ts"] = events_sorted.groupby(["user_id","function_id"])['timestamp'].shift(1)
    events_sorted["revisit"] = ((events_sorted["timestamp"] - events_sorted["prev_ts"]).dt.total_seconds() <= 3600).fillna(False)
    return_count = events_sorted.groupby(["user_id","function_id"])['revisit'].sum().rename("return_count")
    
    # 마지막 사용 시점 기준 일수
    max_ts = events_df["timestamp"].max()
    last_access_days = (max_ts - grp["timestamp"].max()).dt.days.rename("last_access_days")
    
    uf = pd.concat([entry_count, click_rate, visit_duration, return_count, last_access_days], axis=1).reset_index()
    
    print("   타깃 변수 생성 중...")
    
    # 노출 여부 타깃: 클릭률/체류/재방문 가중 합이 임계치 이상
    score = (uf["click_rate"]*0.5 + (uf["visit_duration"]/uf["visit_duration"].max())*0.3 + (uf["return_count"]/max(1,uf["return_count"].max()))*0.2)
    uf["exposure_label"] = (score > score.quantile(0.5)).astype(int)
    
    # UI 유형 타깃: 과거 관찰된 component_type 최빈값
    ui_type = events_df.groupby(["user_id","function_id"])['component_type'].agg(lambda x: x.value_counts().index[0]).rename("ui_type_label")
    
    # 그룹/소제목 타깃: 기능 메타로부터(최빈 cluster/label)
    cluster = events_df.groupby(["user_id","function_id"])['service_cluster'].agg(lambda x: x.value_counts().index[0]).rename("service_cluster_label")
    label = events_df.groupby(["user_id","function_id"])['label'].agg(lambda x: x.value_counts().index[0]).rename("label_text")
    
    # 배치 순서 타깃: 사용자 내 기능별 점수 내림차순 순위(1 = 최상단)
    uf["rank_score"] = score
    uf["rank_label"] = uf.groupby("user_id")["rank_score"].rank(ascending=False, method="first")
    
    targets = uf.merge(ui_type.reset_index(), on=["user_id","function_id"], how="left") \
                .merge(cluster.reset_index(), on=["user_id","function_id"], how="left") \
                .merge(label.reset_index(), on=["user_id","function_id"], how="left")
    
    # 사용자 레벨 피처 결합 + 메타 정보(나이/디바이스 등)
    meta = events_df.groupby("user_id")[["age_group","is_senior","device_type"]].agg(lambda x: x.iloc[0]).reset_index()
    feat = uf.merge(user_df, on="user_id", how="left") 
    feat = feat.merge(meta, on="user_id", how="left")
    
    # 타깃 생성 및 병합
    feat = feat.merge(targets.drop(columns=["entry_count","click_rate","visit_duration","return_count","last_access_days","rank_score"]),
                      on=["user_id","function_id"], how="left")
    
    # CSV 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    features_path = f"data/processed/features_{timestamp}.csv"
    feat.to_csv(features_path, index=False)
    
    print(f"피처 생성 완료: {features_path}")
    print(f"   총 {len(feat)}개 샘플, {len(feat.columns)}개 피처")
    
    return feat

features_df = build_features(events_df)

# 4. 모델 학습
print("4. 모델 학습 중...")

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

def train_model(df, target_col, model_type, model_name, drop_cols=None):
    """모델 학습 함수"""
    print(f"   {model_name} 모델 학습 중...")
    
    X, y = preprocess_features(df, target_col, drop_cols)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if model_type != "regression" else None)
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    
    # LightGBM 4.x 호환 파라미터
    params = model_config["lgbm"][model_name].copy()
    num_boost_round = params.pop("num_boost_round")
    early_stopping_rounds = params.pop("early_stopping_rounds")
    
    if model_type == "multiclass":
        params["num_class"] = len(y.unique())
    
    # callbacks 설정
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=50)
    ]
    
    model = lgb.train(params, dtrain, num_boost_round=num_boost_round,
                      valid_sets=[dtrain, dval], valid_names=["train","valid"],
                      callbacks=callbacks)
    
    # 모델 저장
    model_path = f"data/models/{model_name}.joblib"
    import joblib
    joblib.dump(model, model_path)
    
    print(f"   {model_name} 모델 학습 완료: {model_path}")
    return model

# 각 모델 학습
exposure_model = train_model(features_df, "exposure_label", "binary", "exposure", 
                            ["ui_type_label","service_cluster_label","label_text","rank_label"])

ui_type_model = train_model(features_df, "ui_type_label", "multiclass", "ui_type", 
                           ["exposure_label","rank_label"])

group_label_model = train_model(features_df, "service_cluster_label", "multiclass", "group_label", 
                               ["exposure_label","rank_label"])

rank_model = train_model(features_df, "rank_label", "regression", "rank", 
                        ["exposure_label","ui_type_label","service_cluster_label","label_text"])

# 5. 추론 및 결과 생성
print("5. 추론 및 결과 생성 중...")

def predict_and_generate_output(features_df, models, allowed_types):
    """추론 및 JSON 출력 생성"""
    print("   추론 실행 중...")
    
    # 입력 데이터 준비
    drop_cols = ["exposure_label","ui_type_label","service_cluster_label","label_text","rank_label"]
    X = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns])
    
    # 피처 전처리
    X_processed, _ = preprocess_features(features_df, "exposure_label", drop_cols)
    
    # 예측
    exp_prob = models['exposure'].predict(X_processed)
    exp_pred = (exp_prob > 0.5).astype(int)
    
    ui_raw = models['ui_type'].predict(X_processed)
    ui_idx = np.argmax(ui_raw, axis=1)
    ui_label = [model_config["ui"]["allowed_component_types"][i % len(model_config["ui"]["allowed_component_types")] for i in ui_idx]
    
    grp_raw = models['group_label'].predict(X_processed)
    grp_idx = np.argmax(grp_raw, axis=1)
    grp_label = [list(model_config["service_clusters"]["labels"].keys())[i % len(model_config["service_clusters"]["labels"])] for i in grp_idx]
    
    rank_pred = models['rank'].predict(X_processed)
    
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
        
        items = []
        for _, row in udf.iterrows():
            if int(row["include"]) == 0:
                continue
            
            # 허용된 UI 타입만 사용
            ui_type = str(row["ui"])
            if ui_type not in allowed_types:
                ui_type = allowed_types[0]
            
            items.append({
                "function_id": row["function_id"],
                "include": True,
                "component_type": ui_type,
                "service_cluster": str(row["grp"]),
                "label": model_config["service_clusters"]["labels"].get(str(row["grp"]), str(row.get("label_text",""))),
                "order": float(row["order"])
            })
        
        # 순서대로 정렬
        items = sorted(items, key=lambda x: x["order"])
        
        outputs.append({
            "user_id": user_id,
            "home": {
                "layout_density": layout,
                "functions": items
            }
        })
    
    # JSON 저장
    output_path = "data/outputs/ui_home_outputs.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    
    print(f"추론 완료: {output_path}")
    print(f"   총 {len(outputs)}명 사용자에 대한 홈 화면 구성 생성")
    
    return outputs

# 추론 실행
models = {
    'exposure': exposure_model,
    'ui_type': ui_type_model,
    'group_label': group_label_model,
    'rank': rank_model
}

allowed_types = ["card", "list_item", "banner", "icon"]
results = predict_and_generate_output(features_df, models, allowed_types)

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
    print(f"사용자: {sample_user['user_id']}")
    print(f"레이아웃 밀도: {sample_user['home']['layout_density']}")
    print(f"기능 수: {len(sample_user['home']['functions'])}")
    if sample_user['home']['functions']:
        sample_func = sample_user['home']['functions'][0]
        print(f"첫 번째 기능: {sample_func['function_id']} ({sample_func['component_type']})")
