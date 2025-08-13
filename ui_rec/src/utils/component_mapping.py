"""
UI 컴포넌트 타입 결정 시스템
"""

import numpy as np

def extract_function_features_for_model(functions):
    """
    모델 추론을 위한 피처 추출
    
    Args:
        functions (List[Dict]): 기능 리스트
        
    Returns:
        Dict: 모델 입력용 피처
    """
    if not functions:
        return {}
    
    # 1. 기본 통계
    function_count = len(functions)
    
    # 2. 서비스 클러스터 분석
    clusters = [f.get("service_cluster", "unknown") for f in functions]
    cluster_counts = {}
    for cluster in set(clusters):
        cluster_counts[f"cluster_{cluster}"] = clusters.count(cluster)
    
    # 3. 기능별 특성
    exposure_avg = np.mean([f.get("exposure", 0.0) for f in functions])
    rank_avg = np.mean([f.get("rank", 0.0) for f in functions])
    
    # 4. 피처 벡터 생성
    features = {
        "function_count": function_count,
        "exposure_avg": exposure_avg,
        "rank_avg": rank_avg,
        **cluster_counts
    }
    
    return features


def predict_component_type_with_model(functions, user_profile, model):
    """
    모델을 사용해서 component_type 직접 예측
    
    Args:
        functions (List[Dict]): 기능 리스트
        user_profile (Dict): 사용자 프로필
        model: 학습된 component_type 예측 모델
        
    Returns:
        str: 예측된 component_type
    """
    
    # 1. 피처 추출
    function_features = extract_function_features_for_model(functions)
    user_features = {
        "is_senior": user_profile.get("is_senior", False)
    }
    
    # 2. 피처 결합
    combined_features = {**function_features, **user_features}
    
    # 3. 모델 예측
    component_type_pred = model.predict([list(combined_features.values())])
    
    return component_type_pred[0]


def get_component_type(functions, user_profile, model=None):
    """
    컴포넌트 타입 결정 (모델 우선, fallback 규칙)
    
    Args:
        functions (List[Dict]): 기능 리스트
        user_profile (Dict): 사용자 프로필 (is_senior, age_group 등)
        model: 학습된 component_type 예측 모델 (선택사항)
        
    Returns:
        str: 최적의 컴포넌트 타입
    """
    
    # 모델이 있으면 모델 사용
    if model is not None:
        try:
            return predict_component_type_with_model(functions, user_profile, model)
        except Exception as e:
            print(f"모델 예측 실패, fallback 규칙 사용: {e}")
    
    # 모델이 없거나 실패시 기존 규칙 사용
    function_nature = analyze_function_nature(functions)
    user_behavior = analyze_user_behavior(user_profile)
    component_type = make_smart_decision(function_nature, user_behavior)
    
    return component_type


def analyze_function_nature(functions):
    """기능의 성격 분석"""
    if not functions:
        return {"count": 0, "main_cluster": "unknown", "type": "unknown"}
    
    # 기능 개수
    function_count = len(functions)
    
    # 서비스 클러스터 분석
    clusters = [f.get("service_cluster", "unknown") for f in functions]
    main_cluster = max(set(clusters), key=clusters.count)
    
    # 기능 타입 분석
    if function_count == 1:
        function_type = "single_function"
    elif function_count <= 3:
        function_type = "few_functions"
    elif function_count <= 6:
        function_type = "medium_functions"
    else:
        function_type = "many_functions"
    
    return {
        "count": function_count,
        "main_cluster": main_cluster,
        "type": function_type
    }


def analyze_user_behavior(user_profile):
    """사용자 행동 패턴 분석"""
    is_senior = user_profile.get("is_senior", False)
    
    if is_senior:
        return "senior_preference"
    else:
        return "normal_preference"


def make_smart_decision(function_nature, user_behavior):
    decision_matrix = {
        "senior_preference": {
            "single_function": "s_list_single",
            "few_functions": "s_list_multi",
            "medium_functions": "s_grid_multi",
            "many_functions": "s_grid_full"
        },
        "normal_preference": {
            "single_function": "list_single",
            "few_functions": "list_multi",
            "medium_functions": "grid_multi",
            "many_functions": "grid_full"
        }
    }
    
    # 기본 결정
    base_component = decision_matrix.get(user_behavior, {}).get(
        function_nature.get("type", "medium_functions"), 
        "grid_multi"
    )
    
    return base_component


def train_component_type_model(training_data):
    """
    component_type 예측 모델 학습
    
    Args:
        training_data (List[Dict]): 학습 데이터
            - functions: 기능 리스트
            - user_profile: 사용자 프로필
            - target_component_type: 실제 사용된 component_type
            
    Returns:
        trained_model: 학습된 모델
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        # 1. 피처와 타겟 분리
        X = []
        y = []
        
        for data in training_data:
            features = extract_function_features_for_model(data["functions"])
            user_features = {
                "is_senior": data["user_profile"].get("is_senior", False)
            }
            combined_features = {**features, **user_features}
            
            X.append(list(combined_features.values()))
            y.append(data["target_component_type"])
        
        # 2. 라벨 인코딩
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # 3. 학습/검증 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # 4. 모델 학습
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 5. 성능 평가
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Component Type 모델 학습 완료")
        print(f"Train Score: {train_score:.3f}")
        print(f"Test Score: {test_score:.3f}")
        
        return model, le
        
    except ImportError:
        print("scikit-learn이 설치되지 않았습니다. 규칙 기반 방식만 사용 가능합니다.")
        return None, None
