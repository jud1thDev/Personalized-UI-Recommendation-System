import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from ..utils.io import read_yaml, ensure_dir, latest_file, save_model
import joblib

# 코랩과 로컬 환경 모두 지원
try:
    CFG_DATA = read_yaml("ui_rec/config/data.yaml")
    CFG_MODEL = read_yaml("ui_rec/config/model.yaml")
except FileNotFoundError:
    # 코랩 환경에서 상대 경로로 시도
    try:
        CFG_DATA = read_yaml("config/data.yaml")
        CFG_MODEL = read_yaml("config/model.yaml")
    except FileNotFoundError:
        # 기본값 사용
        CFG_DATA = {
            "paths": {
                "processed_dir": "data/processed",
                "models_dir": "data/models"
            },
            "files": {
                "features_pattern": "features_*.csv"
            }
        }
        CFG_MODEL = {
            "lgbm": {
                "exposure": {"objective": "binary", "metric": "auc"},
                "ui_type": {"objective": "multiclass", "metric": "multi_logloss"},
                "service_cluster": {"objective": "multiclass", "metric": "multi_logloss"},
                "rank": {"objective": "regression", "metric": "rmse"}
            }
        }

PROC_DIR = CFG_DATA["paths"]["processed_dir"]
MODELS_DIR = CFG_DATA["paths"]["models_dir"]

# 학습/추론에서 항상 제거할 컬럼
ALWAYS_DROP = ["user_id", "function_id"]

def load_latest_features() -> pd.DataFrame:
    path = latest_file(CFG_DATA["files"]["features_pattern"], PROC_DIR)
    if not path:
        raise FileNotFoundError("No features file. Run build_features.py first.")
    return pd.read_csv(path)


def _encode_types(X: pd.DataFrame, categorical_cols=None, boolean_cols=None) -> pd.DataFrame:
    """object/category -> codes, bool -> int8"""
    X = X.copy()

    # 명시된 카테고리 컬럼 우선 적용, 아니면 dtype 기반 자동 탐지
    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("category").cat.codes

    # 이미 category dtype인 것들도 codes로
    for c in X.select_dtypes(include=["category"]).columns:
        X[c] = X[c].cat.codes

    # 불린 컬럼
    if boolean_cols is None:
        boolean_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for c in boolean_cols:
        if c in X.columns:
            X[c] = X[c].astype("int8")

    return X


def split_xy(df: pd.DataFrame, target: str, drop_cols=None):
    """학습용 분리: ID/타겟/추가 drop 제외 + 타입 인코딩"""
    if drop_cols is None:
        drop_cols = []
    drop_cols = list(set(drop_cols + ALWAYS_DROP + [target]))

    X = df.drop(columns=drop_cols)
    X = _encode_types(X)

    y = df[target]
    if y.dtype == "object":
        y = y.astype("category").cat.codes
    return X, y


def get_feature_mapping(df: pd.DataFrame, target: str, drop_cols=None):
    """학습 시 저장할 매핑(열 순서/타입 정보)"""
    if drop_cols is None:
        drop_cols = []
    drop_cols = list(set(drop_cols + ALWAYS_DROP + [target]))

    X = df.drop(columns=drop_cols)
    mapping = {
        "categorical_columns": X.select_dtypes(include=["object"]).columns.tolist(),
        "boolean_columns": X.select_dtypes(include=["bool"]).columns.tolist(),
        "column_order": X.columns.tolist(),
        "feature_count": X.shape[1],
    }
    return mapping


def align_features_with_mapping(X: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """추론 시 학습과 동일한 열/타입 정렬"""
    X2 = X.copy()

    # 누락된 열 0으로 채우기
    for col in mapping["column_order"]:
        if col not in X2.columns:
            X2[col] = 0

    # 여분 열 제거 + 순서 강제
    X2 = X2[mapping["column_order"]]

    # 타입 인코딩 동일 적용
    X2 = _encode_types(X2, mapping["categorical_columns"], mapping["boolean_columns"])
    return X2


def apply_feature_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """(이전 호환용) 저장된 매핑 기반으로 열 정렬 + 기본 타입 설정"""
    X = df[mapping['column_order']].copy()
    for col in mapping['categorical_columns']:
        if col in X.columns:
            X[col] = X[col].astype("category")
    for col in mapping['boolean_columns']:
        if col in X.columns:
            X[col] = X[col].astype("int8")
    return X


def train_binary(df: pd.DataFrame, target: str, model_name: str) -> str:
    params = CFG_MODEL["lgbm"]["exposure"].copy()
    drop_cols = ["ui_type_label","service_cluster_label","rank_label"]
    X, y = split_xy(df, target, drop_cols)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)

    num_boost_round = params.pop("num_boost_round")
    early_stopping_rounds = params.pop("early_stopping_rounds")
    
    # LightGBM 4.x 호환성을 위한 callbacks 설정
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=50)
    ]

    model = lgb.train(params, dtrain, num_boost_round=num_boost_round,
                      valid_sets=[dtrain, dval], valid_names=["train","valid"],
                      callbacks=callbacks)
    
    # 피처 매핑 정보도 함께 저장
    feature_mapping = get_feature_mapping(df, target, drop_cols)
    model_data = {
        'model': model,
        'feature_mapping': feature_mapping
    }
    
    out_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    save_model(model_data, out_path)
    return out_path


def train_multiclass(df: pd.DataFrame, target: str, model_key: str, model_name: str) -> str:
    params = CFG_MODEL["lgbm"][model_key].copy()
    classes = df[target].astype("category").cat.categories.tolist()
    params["num_class"] = len(classes)

    drop_cols = ["exposure_label","rank_label","ui_type_label","service_cluster_label"]
    X, y = split_xy(df, target, drop_cols)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)

    num_boost_round = params.pop("num_boost_round")
    early_stopping_rounds = params.pop("early_stopping_rounds")
    
    # LightGBM 4.x 호환성을 위한 callbacks 설정
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=50)
    ]

    model = lgb.train(params, dtrain, num_boost_round=num_boost_round,
                      valid_sets=[dtrain, dval], valid_names=["train","valid"],
                      callbacks=callbacks)

    # 피처 매핑 정보도 함께 저장
    feature_mapping = get_feature_mapping(df, target, drop_cols)
    model_data = {
        'model': model,
        'classes': classes,
        'feature_mapping': feature_mapping
    }

    out_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    save_model(model_data, out_path)
    return out_path


def train_regression(df: pd.DataFrame, target: str, model_name: str) -> str:
    params = CFG_MODEL["lgbm"]["rank"].copy()
    drop_cols = ["exposure_label","ui_type_label","service_cluster_label","rank_label"]
    X, y = split_xy(df, target, drop_cols)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)

    num_boost_round = params.pop("num_boost_round")
    early_stopping_rounds = params.pop("early_stopping_rounds")
    
    # LightGBM 4.x 호환성을 위한 callbacks 설정
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=50)
    ]

    model = lgb.train(params, dtrain, num_boost_round=num_boost_round,
                      valid_sets=[dtrain, dval], valid_names=["train","valid"],
                      callbacks=callbacks)

    # 피처 매핑 정보도 함께 저장
    feature_mapping = get_feature_mapping(df, target, drop_cols)
    model_data = {
        'model': model,
        'feature_mapping': feature_mapping
    }

    out_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    save_model(model_data, out_path)
    return out_path 


def split_xy_colab(df: pd.DataFrame, target: str, drop_cols=None):
    """Colab에서 간편 사용: ID 자동 제거 + 인코딩"""
    if drop_cols is None:
        drop_cols = []
    drop_cols = list(set(drop_cols + ALWAYS_DROP + [target]))

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target].copy()

    X = _encode_types(X)
    return X, y


def load_model_colab(model_path: str):
    """(이전 호환용) 모델만 꺼내기"""
    model_data = joblib.load(model_path)
    if isinstance(model_data, dict) and 'model' in model_data:
        return model_data['model']
    return model_data


def load_model_and_mapping(path: str):
    data = joblib.load(path)
    if isinstance(data, dict) and "model" in data:
        return data["model"], data.get("feature_mapping")
    return data, None


def _predict_with_mapping(model_path: str, X: pd.DataFrame):
    model, mapping = load_model_and_mapping(model_path)
    if mapping is not None:
        Xa = align_features_with_mapping(X, mapping)
    else:
        Xa = _encode_types(X)  
    return model.predict(Xa)


def predict_exposure_colab(model_path: str, X: pd.DataFrame):
    return _predict_with_mapping(model_path, X)


def predict_ui_type_colab(model_path: str, X: pd.DataFrame):
    return _predict_with_mapping(model_path, X)


def predict_service_cluster_colab(model_path: str, X: pd.DataFrame):
    return _predict_with_mapping(model_path, X)


def predict_rank_colab(model_path: str, X: pd.DataFrame):
    return _predict_with_mapping(model_path, X)


def predict_all_models_colab(model_dir: str, X: pd.DataFrame):
    return {
        "exposure":        predict_exposure_colab(os.path.join(model_dir, "exposure.joblib"), X),
        "ui_type":         predict_ui_type_colab(os.path.join(model_dir, "ui_type.joblib"), X),
        "service_cluster": predict_service_cluster_colab(os.path.join(model_dir, "service_cluster.joblib"), X),
        "rank":            predict_rank_colab(os.path.join(model_dir, "rank.joblib"), X),
    } 