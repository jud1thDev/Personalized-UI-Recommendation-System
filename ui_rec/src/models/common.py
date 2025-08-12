import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from ..utils.io import read_yaml, ensure_dir, latest_file, save_model
import joblib

CFG_DATA = read_yaml("ui_rec/config/data.yaml")
CFG_MODEL = read_yaml("ui_rec/config/model.yaml")
PROC_DIR = CFG_DATA["paths"]["processed_dir"]
MODELS_DIR = CFG_DATA["paths"]["models_dir"]


def load_latest_features() -> pd.DataFrame:
    path = latest_file(CFG_DATA["files"]["features_pattern"], PROC_DIR)
    if not path:
        raise FileNotFoundError("No features file. Run build_features.py first.")
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame, target: str, drop_cols=None):
    if drop_cols is None:
        drop_cols = []
    X = df.drop(columns=[target] + drop_cols)
    
    # 모든 object 컬럼을 범주형 코드로 변환
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category").cat.codes
    
    # bool은 정수로 변환
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype("int8")
    
    # category 컬럼을 정수로 변환
    for col in X.select_dtypes(include=["category"]).columns:
        X[col] = X[col].cat.codes
    
    y = df[target]
    
    # 타겟 변수가 문자열인 경우 숫자로 인코딩
    if y.dtype == "object":
        y = y.astype("category").cat.codes
    
    return X, y


def get_feature_mapping(df: pd.DataFrame, target: str, drop_cols=None):
    """피처 매핑 정보를 반환하여 추론 시 동일한 전처리 적용"""
    if drop_cols is None:
        drop_cols = []
    X = df.drop(columns=[target] + drop_cols)
    
    mapping = {
        'categorical_columns': X.select_dtypes(include=["object"]).columns.tolist(),
        'boolean_columns': X.select_dtypes(include=["bool"]).columns.tolist(),
        'column_order': X.columns.tolist(),
        'feature_count': len(X.columns)
    }
    
    return mapping


def apply_feature_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """저장된 피처 매핑을 사용하여 동일한 전처리 적용"""
    X = df[mapping['column_order']].copy()
    
    # 범주형 변환
    for col in mapping['categorical_columns']:
        if col in X.columns:
            X[col] = X[col].astype("category")
    
    # 불린 변환
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
    """Colab 환경에서 사용하는 데이터 분할 함수"""
    if drop_cols is None:
        drop_cols = []
    
    # 타겟 컬럼과 제거할 컬럼들을 제외한 피처 데이터
    feature_cols = [col for col in df.columns if col not in [target] + drop_cols]
    X = df[feature_cols].copy()
    y = df[target].copy()
    
    # 범주형 변수 인코딩
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category").cat.codes
    
    # bool은 정수로 변환
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype("int8")
    
    return X, y


def load_model_colab(model_path: str):
    """Colab 환경에서 모델을 로드하는 함수"""
    model_data = joblib.load(model_path)
    if isinstance(model_data, dict) and 'model' in model_data:
        return model_data['model']
    return model_data


def predict_exposure_colab(model_path: str, X: pd.DataFrame):
    """노출 예측을 수행하는 함수"""
    model = load_model_colab(model_path)
    return model.predict(X)


def predict_ui_type_colab(model_path: str, X: pd.DataFrame):
    """UI 타입 예측을 수행하는 함수"""
    model = load_model_colab(model_path)
    return model.predict(X)


def predict_service_cluster_colab(model_path: str, X: pd.DataFrame):
    """서비스 클러스터 예측을 수행하는 함수"""
    model = load_model_colab(model_path)
    return model.predict(X)


def predict_rank_colab(model_path: str, X: pd.DataFrame):
    """순위 예측을 수행하는 함수"""
    model = load_model_colab(model_path)
    return model.predict(X)


def predict_all_models_colab(model_dir: str, X: pd.DataFrame):
    """모든 모델을 사용하여 예측을 수행하는 함수"""
    # 예측 전에 범주형 변수 인코딩
    X_encoded = X.copy()
    
    # 범주형 변수 인코딩
    for col in X_encoded.select_dtypes(include=["object"]).columns:
        X_encoded[col] = X_encoded[col].astype("category").cat.codes
    
    # bool은 정수로 변환
    for col in X_encoded.select_dtypes(include=["bool"]).columns:
        X_encoded[col] = X_encoded[col].astype("int8")
    
    results = {}
    
    # 각 모델별 예측 수행
    results['exposure'] = predict_exposure_colab(f"{model_dir}/exposure.joblib", X_encoded)
    results['ui_type'] = predict_ui_type_colab(f"{model_dir}/ui_type.joblib", X_encoded)
    results['service_cluster'] = predict_service_cluster_colab(f"{model_dir}/service_cluster.joblib", X_encoded)
    results['rank'] = predict_rank_colab(f"{model_dir}/rank.joblib", X_encoded)
    
    return results 