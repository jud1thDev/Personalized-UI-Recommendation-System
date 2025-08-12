"""
UI 컴포넌트 그룹화 및 라벨링 모듈
Hugging Face AI 모델을 사용해서 자연스러운 라벨 자동 생성
"""

from typing import List, Dict
from collections import Counter
import random


def create_ui_component_groups(functions: List[Dict], user_name: str = "") -> List[Dict]:
    """UI 컴포넌트 타입별로 기능들을 그룹화하고 AI 기반 라벨 생성"""
    
    if len(functions) <= 2:
        # 1-2개 기능은 추천 기능 그룹으로 묶어버림
        label = f"{user_name}님의 추천 기능" if user_name else "추천 기능"
        return [{"label": label, "functions": functions, "component_type": "mixed", "function_count": len(functions)}]
    
    # UI 컴포넌트 타입별로 그룹화
    component_groups = {}
    for func in functions:
        comp_type = func.get("component_type", "card")
        if comp_type not in component_groups:
            component_groups[comp_type] = []
        component_groups[comp_type].append(func)
    
    # AI 라벨 생성기를 한 번만 생성 (싱글톤 패턴으로 중복 방지)
    ai_generator = None
    try:
        from ..utils.ai_label_generator import create_ai_label_generator
        ai_generator = create_ai_label_generator()
    except Exception as e:
        print(f"AI label generator initialization failed: {e}")
        ai_generator = None
    
    # 각 컴포넌트 타입별로 그룹 생성
    groups = []
    for comp_type, funcs in component_groups.items():
        if len(funcs) == 0:
            continue
            
        # AI 기반 라벨 생성 (3개 이상 기능)
        if len(funcs) >= 3 and ai_generator:
            try:
                label = ai_generator.generate_label(funcs, comp_type)
            except Exception as e:
                print(f"AI label generation failed for {comp_type}: {e}")
                label = generate_simple_label(funcs, comp_type, user_name)
        else:
            # 1-2개 기능은 추천 기능 그룹
            label = generate_simple_label(funcs, comp_type, user_name)
        
        groups.append({
            "label": label,
            "functions": funcs,
            "component_type": comp_type,
            "function_count": len(funcs),
            "ui_style": get_ui_component_style(comp_type)
        })
    
    # 그룹이 하나뿐이면 더 세분화
    if len(groups) == 1 and len(functions) > 4:
        # 기능 개수로 2-3개 그룹으로 분할
        mid_point = len(functions) // 2
        
        # AI 라벨 생성 시도
        label1 = "추천 기능"
        label2 = "추천 기능"
        
        if ai_generator:
            try:
                label1 = ai_generator.generate_label(functions[:mid_point], "mixed")
                label2 = ai_generator.generate_label(functions[mid_point:], "mixed")
            except Exception as e:
                print(f"AI label generation for split groups failed: {e}")
        
        return [
            {
                "label": label1,
                "functions": functions[:mid_point],
                "component_type": "mixed",
                "function_count": len(functions[:mid_point]),
                "ui_style": get_ui_component_style("mixed")
            },
            {
                "label": label2,
                "functions": functions[mid_point:],
                "component_type": "mixed",
                "function_count": len(functions[mid_point:]),
                "ui_style": get_ui_component_style("mixed")
            }
        ]
    
    return groups


def generate_simple_label(functions: List[Dict], component_type: str, user_name: str = "") -> str:
    """간단한 라벨 생성 (AI 실패 시 fallback)"""
    
    # 서비스 클러스터 분석
    cluster_counts = Counter()
    for func in functions:
        cluster = func.get("service_cluster", "unknown")
        cluster_counts[cluster] += 1
    
    # 주요 서비스 클러스터 파악
    if cluster_counts:
        main_cluster = cluster_counts.most_common(1)[0][0]
        return get_simple_fallback_label(main_cluster, user_name)
    
    return f"{user_name}님의 추천 기능" if user_name else "추천 기능"


def get_simple_fallback_label(cluster: str, user_name: str = "") -> str:
    """AI 생성 실패 시 간단한 fallback 라벨"""
    simple_labels = {
        "account": "계정",
        "finance": "금융", 
        "lifestyle": "생활",
        "health": "건강",
        "shopping": "쇼핑",
        "travel": "여행",
        "security": "보안",
        "unknown": "기타"
    }
    
    base_label = simple_labels.get(cluster, cluster)
    return f"{user_name}님의 {base_label}" if user_name else base_label


def get_ui_component_style(component_type: str) -> Dict[str, str]:
    """UI 컴포넌트 타입별 스타일 정보""" # ToDo: 추후 SDK 활용 예정
    
    base_styles = {
        "card": {
            "display": "grid",
            "grid_template_columns": "repeat(auto-fit, minmax(280px, 1fr))",
            "gap": "16px",
            "padding": "20px"
        },
        "banner": {
            "display": "flex",
            "flex_direction": "column",
            "gap": "12px",
            "padding": "16px"
        },
        "list_item": {
            "display": "flex", 
            "flex_direction": "column",
            "gap": "8px",
            "padding": "12px"
        },
        "icon": {
            "display": "grid",
            "grid_template_columns": "repeat(auto-fit, minmax(80px, 1fr))",
            "gap": "12px",
            "padding": "16px"
        },
        "grid_item": {
            "display": "grid",
            "grid_template_columns": "repeat(auto-fit, minmax(200px, 1fr))",
            "gap": "16px",
            "padding": "20px"
        },
        "mixed": {
            "display": "flex",
            "flex_direction": "column",
            "gap": "16px",
            "padding": "20px"
        }
    }
    
    return base_styles.get(component_type, base_styles["mixed"])
