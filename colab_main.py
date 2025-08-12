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
        "allowed_types": ["card", "list_item", "banner", "icon", "grid_item"],
        "default_type": "card"
    },
    "service_clusters": {
        "mapping": {
            "f001": "account", "f002": "account", "f003": "account",
            "f004": "finance", "f005": "finance", "f006": "finance",
            "f007": "lifestyle", "f008": "lifestyle", "f009": "lifestyle",
            "f010": "health", "f011": "health", "f012": "health",
            "f013": "shopping", "f014": "shopping", "f015": "shopping",
            "f016": "travel", "f017": "travel", "f018": "travel",
            "f019": "security", "f020": "security"
        },
        "labels": {
            "account": "계정 관리",
            "finance": "금융 서비스", 
            "lifestyle": "라이프스타일",
            "health": "건강 관리",
            "shopping": "쇼핑",
            "travel": "여행/교통",
            "security": "보안"
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
print("\n2. 모의 데이터 생성 중...")

def generate_users(n_users=20):
    """사용자 데이터 생성"""
    users = []
    for i in range(n_users):
        user = {
            "user_id": f"U{i:05d}",
            "age": random.randint(20, 80),
            "gender": random.choice(["M", "F"]),
            "senior": random.choice([True, False]),
            "premium": random.choice([True, False]),
            "location": random.choice(["Seoul", "Busan", "Incheon", "Daegu", "Daejeon"])
        }
        users.append(user)
    return users

def generate_functions():
    """기능 데이터 생성"""
    functions = []
    for i in range(1, 21):
        func = {
            "function_id": f"f{i:03d}",
            "name": f"Function {i}",
            "category": random.choice(["account", "finance", "lifestyle", "health", "shopping", "travel", "security"]),
            "complexity": random.randint(1, 5)
        }
        functions.append(func)
    return functions

def simulate_events(users, funcs, days=7):
    """사용자 이벤트 시뮬레이션"""
    events = []
    start_date = datetime.now() - timedelta(days=days)
    
    for user in users:
        # 사용자별 세션 수 (1-5개)
        n_sessions = random.randint(1, 5)
        
        for session in range(n_sessions):
            session_start = start_date + timedelta(
                days=random.randint(0, days),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # 세션당 기능 사용 수 (1-8개)
            n_functions = random.randint(1, 8)
            used_functions = random.sample(funcs, min(n_functions, len(funcs)))
            
            for func in used_functions:
                # 기능별 사용 패턴
                n_clicks = random.randint(1, 10)
                visit_duration = random.randint(10, 300)  # 초
                
                for click in range(n_clicks):
                    event = {
                        "user_id": user["user_id"],
                        "function_id": func["function_id"],
                        "timestamp": session_start + timedelta(
                            minutes=click * random.randint(1, 5)
                        ),
                        "event_type": "click",
                        "session_id": f"{user['user_id']}_s{session}",
                        "click_count": click + 1,
                        "visit_duration": visit_duration,
                        "return_count": random.randint(0, 3)
                    }
                    events.append(event)
    
    return events

# 데이터 생성 실행
users = generate_users(20)
funcs = generate_functions()
events = simulate_events(users, funcs, 7)

# CSV로 저장
events_df = pd.DataFrame(events)
events_df.to_csv('data/raw/events.csv', index=False, encoding='utf-8')

print(f"모의 데이터 생성 완료: {len(users)}명 사용자, {len(events)}개 이벤트")

# 3. 피처 생성
print("\n3. 피처 생성 중...")

def build_features_colab():
    """피처 생성 (코랩용)"""
    # 이벤트 데이터 로드
    events_df = pd.read_csv('data/raw/events.csv')
    
    # 사용자별 피처 생성
    user_features = []
    
    for user_id in events_df['user_id'].unique():
        user_events = events_df[events_df['user_id'] == user_id]
        
        # 기본 피처
        features = {
            'user_id': user_id,
            'total_events': len(user_events),
            'unique_functions': user_events['function_id'].nunique(),
            'total_sessions': user_events['session_id'].nunique(),
            'avg_click_count': user_events.groupby('function_id')['click_count'].max().mean(),
            'avg_visit_duration': user_events['visit_duration'].mean(),
            'avg_return_count': user_events['return_count'].mean()
        }
        
        # 시간대별 피처
        user_events['hour'] = pd.to_datetime(user_events['timestamp']).dt.hour
        features['morning_usage'] = len(user_events[user_events['hour'].between(6, 11)])
        features['afternoon_usage'] = len(user_events[user_events['hour'].between(12, 17)])
        features['evening_usage'] = len(user_events[user_events['hour'].between(18, 23)])
        features['night_usage'] = len(user_events[user_events['hour'].between(0, 5)])
        
        # 세션 패턴
        session_durations = []
        for session_id in user_events['session_id'].unique():
            session_events = user_events[user_events['session_id'] == session_id]
            if len(session_events) > 1:
                start_time = pd.to_datetime(session_events['timestamp'].min())
                end_time = pd.to_datetime(session_events['timestamp'].max())
                duration = (end_time - start_time).total_seconds() / 60  # 분
                session_durations.append(duration)
        
        features['avg_session_duration'] = np.mean(session_durations) if session_durations else 0
        
        user_features.append(features)
    
    # 기능별 피처 생성
    function_features = []
    
    for function_id in events_df['function_id'].unique():
        func_events = events_df[events_df['function_id'] == function_id]
        
        features = {
            'function_id': function_id,
            'total_clicks': len(func_events),
            'unique_users': func_events['user_id'].nunique(),
            'avg_click_per_user': len(func_events) / func_events['user_id'].nunique(),
            'avg_visit_duration': func_events['visit_duration'].mean(),
            'return_rate': (func_events['return_count'] > 0).mean()
        }
        
        function_features.append(features)
    
    # 사용자-기능 조합 피처
    user_function_features = []
    
    for user_id in events_df['user_id'].unique():
        for function_id in events_df['function_id'].unique():
            user_func_events = events_df[
                (events_df['user_id'] == user_id) & 
                (events_df['function_id'] == function_id)
            ]
            
            if len(user_func_events) > 0:
                features = {
                    'user_id': user_id,
                    'function_id': function_id,
                    'click_count': len(user_func_events),
                    'visit_duration': user_func_events['visit_duration'].sum(),
                    'return_count': user_func_events['return_count'].sum(),
                    'last_access_days': (datetime.now() - pd.to_datetime(user_func_events['timestamp'].max())).days
                }
                
                # 세션 관련 피처
                sessions = user_func_events['session_id'].unique()
                features['session_count'] = len(sessions)
                features['avg_clicks_per_session'] = len(user_func_events) / len(sessions)
                
                user_function_features.append(features)
    
    # 피처 결합
    user_df = pd.DataFrame(user_features)
    function_df = pd.DataFrame(function_features)
    user_function_df = pd.DataFrame(user_function_features)
    
    # 최종 피처 데이터프레임 생성
    final_features = user_function_df.merge(user_df, on='user_id', suffixes=('', '_user'))
    final_features = final_features.merge(function_df, on='function_id', suffixes=('', '_func'))
    
    # 타겟 변수 생성
    final_features['exposure'] = (final_features['click_count'] > 0).astype(int)
    final_features['ui_type'] = np.random.choice(['card', 'list_item', 'banner', 'icon'], size=len(final_features))
    final_features['service_cluster'] = final_features['function_id'].map(model_config['service_clusters']['mapping'])
    final_features['rank'] = np.random.uniform(1, 10, size=len(final_features))
    
    # CSV로 저장
    final_features.to_csv('data/processed/features.csv', index=False, encoding='utf-8')
    
    return final_features

# 피처 생성 실행
features_df = build_features_colab()
print(f"피처 생성 완료: {len(features_df)}개 샘플")

# 4. 모델 학습
print("\n4. 모델 학습 중...")

def split_xy(df, target_col, drop_cols=None):
    """X, y 분리"""
    if drop_cols is None:
        drop_cols = []
    
    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]
    
    # 수치형 컬럼만 선택
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    return X, y

def train_model(X, y, model_type, params):
    """모델 학습"""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "binary":
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
    elif model_type == "multiclass":
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
    else:  # regression
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
    
    return model

# 모델 학습 실행
models = {}

# 1) Exposure 모델 (Binary)
print("  - Exposure 모델 학습 중...")
X_exposure, y_exposure = split_xy(features_df, 'exposure', ['ui_type', 'service_cluster', 'rank'])
exposure_model = train_model(X_exposure, y_exposure, "binary", model_config['lgbm']['exposure'])
models['exposure'] = exposure_model

# 2) UI Type 모델 (Multiclass)
print("  - UI Type 모델 학습 중...")
X_ui_type, y_ui_type = split_xy(features_df, 'ui_type', ['exposure', 'service_cluster', 'rank'])
ui_type_model = train_model(X_ui_type, y_ui_type, "multiclass", model_config['lgbm']['ui_type'])
models['ui_type'] = ui_type_model

# 3) Service Cluster 모델 (Multiclass)
print("  - Service Cluster 모델 학습 중...")
X_service_cluster, y_service_cluster = split_xy(features_df, 'service_cluster', ['exposure', 'ui_type', 'rank'])
service_cluster_model = train_model(X_service_cluster, y_service_cluster, "multiclass", model_config['lgbm']['service_cluster'])
models['service_cluster'] = service_cluster_model

# 4) Rank 모델 (Regression)
print("  - Rank 모델 학습 중...")
X_rank, y_rank = split_xy(features_df, 'rank', ['exposure', 'ui_type', 'service_cluster'])
rank_model = train_model(X_rank, y_rank, "regression", model_config['lgbm']['rank'])
models['rank'] = rank_model
    
    # 모델 저장
    import joblib
for name, model in models.items():
    joblib.dump(model, f'data/models/lgbm_{name}.joblib')

print("모델 학습 완료")

# 5. 추론 및 결과 생성
print("\n5. 추론 및 결과 생성 중...")

def get_icon_suggestion(function_id, service_cluster):
    """아이콘 추천"""
    icon_mapping = {
        'account': ['account_circle', 'person', 'security'],
        'finance': ['account_balance', 'credit_card', 'trending_up'],
        'lifestyle': ['home', 'favorite', 'star'],
        'health': ['favorite', 'local_hospital', 'fitness_center'],
        'shopping': ['shopping_cart', 'store', 'local_offer'],
        'travel': ['flight', 'train', 'directions_car'],
        'security': ['security', 'lock', 'verified_user']
    }
    
    icons = icon_mapping.get(service_cluster, ['help'])
    return random.choice(icons)

def create_ui_component_groups(functions, max_per_group=6):
    """UI 컴포넌트 그룹 생성"""
    groups = []
    current_group = []
    
    for func in functions:
        if len(current_group) >= max_per_group:
            # 그룹 완성 및 저장
            if current_group:
                group_label = generate_group_label(current_group)
                groups.append({
                    "label": group_label,
                    "functions": current_group,
                    "component_type": "mixed",
                    "ui_style": {
                        "display": "flex",
                        "flex_direction": "column",
                        "gap": "16px",
                        "padding": "20px"
                    }
                })
            current_group = []
        
        current_group.append(func)
    
    # 마지막 그룹 처리
    if current_group:
        group_label = generate_group_label(current_group)
        groups.append({
            "label": group_label,
            "functions": current_group,
            "component_type": "mixed",
            "ui_style": {
                "display": "flex",
                "flex_direction": "column",
                "gap": "16px",
                "padding": "20px"
            }
        })
    
    return groups

def generate_group_label(functions):
    """그룹 라벨 생성"""
    # 서비스 클러스터별로 그룹화
    cluster_counts = {}
    for func in functions:
        cluster = func.get('service_cluster', 'unknown')
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
    
    # 가장 많은 클러스터를 기준으로 라벨 생성
    if cluster_counts:
        main_cluster = max(cluster_counts, key=cluster_counts.get)
        cluster_label = model_config['service_clusters']['labels'].get(main_cluster, main_cluster)
        
        # 개인화된 라벨 생성
        if len(functions) <= 3:
            return f"{cluster_label} 추천"
        else:
            return f"{cluster_label} 모음"
    
    return "추천 기능"

# 추론 실행
results = []

for user_id in features_df['user_id'].unique():
    user_features = features_df[features_df['user_id'] == user_id]
    
    # 예측
    X_user = user_features.select_dtypes(include=[np.number])
    X_user = X_user.drop(columns=['exposure', 'rank'], errors='ignore')
    
    exposure_pred = exposure_model.predict(X_user)
    ui_type_pred = ui_type_model.predict(X_user)
    service_cluster_pred = service_cluster_model.predict(X_user)
    rank_pred = rank_model.predict(X_user)
    
    # 결과 생성
    exposed_functions = []
    
    for i, (_, row) in enumerate(user_features.iterrows()):
        if exposure_pred[i] == 1:  # 노출 대상
            # 아이콘 추천
            icon_suggestion = get_icon_suggestion(row['function_id'], row['service_cluster'])
            
            function_result = {
                "function_id": row['function_id'],
                "component_type": ui_type_pred[i],
                "service_cluster": service_cluster_pred[i],
                "order": rank_pred[i],
                "icon_suggestion": icon_suggestion
            }
            exposed_functions.append(function_result)
        
        # 순서대로 정렬
    exposed_functions.sort(key=lambda x: x['order'])
    
    # 그룹 생성
    groups = create_ui_component_groups(exposed_functions)
        
    # 사용자별 결과
    user_result = {
            "user_id": user_id,
            "home": {
            "layout_density": "medium",
                "groups": groups
            }
    }
    
    results.append(user_result)

# 결과 저장
with open('data/outputs/ui_home_outputs.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"추론 완료: {len(results)}명 사용자")

# 6. 결과 요약
print("\n" + "=" * 60)
print("Personalized UI Recommendation System 실행 완료!")
print(f"총 사용자: {len(results)}명")
print(f"결과 파일: data/outputs/ui_home_outputs.json")
print("=" * 60)
