"""
UI 컴포넌트 그룹화 및 라벨링 모듈
홈 화면에서 기능들을 UI 컴포넌트 타입별로 그룹화하고 의미있는 라벨을 생성
"""

from typing import List, Dict
from collections import Counter


def create_ui_component_groups(functions: List[Dict], user_name: str = "") -> List[Dict]:
    """UI 컴포넌트 타입별로 기능들을 그룹화하고 의미있는 라벨 생성"""
    
    if len(functions) <= 2:
        # 기능이 적으면 단일 그룹
        label = generate_meaningful_label(functions, user_name)
        return [{"label": label, "functions": functions, "component_type": "mixed"}]
    
    # UI 컴포넌트 타입별로 그룹화
    component_groups = {}
    for func in functions:
        comp_type = func.get("component_type", "card")
        if comp_type not in component_groups:
            component_groups[comp_type] = []
        component_groups[comp_type].append(func)
    
    # 각 컴포넌트 타입별로 그룹 생성
    groups = []
    for comp_type, funcs in component_groups.items():
        if len(funcs) == 0:
            continue
            
        # 의미있는 라벨 생성
        label = generate_meaningful_label(funcs, user_name)
        
        groups.append({
            "label": label,
            "functions": funcs,
            "component_type": comp_type,
            "ui_style": get_ui_component_style(comp_type)
        })
    
    # 그룹이 하나뿐이면 더 세분화
    if len(groups) == 1 and len(functions) > 4:
        # 기능 개수로 2-3개 그룹으로 분할
        mid_point = len(functions) // 2
        
        first_label = generate_meaningful_label(functions[:mid_point], user_name)
        second_label = generate_meaningful_label(functions[mid_point:], user_name)
        
        return [
            {
                "label": first_label,
                "functions": functions[:mid_point],
                "component_type": "mixed",
                "ui_style": get_ui_component_style("mixed")
            },
            {
                "label": second_label,
                "functions": functions[mid_point:],
                "component_type": "mixed", 
                "ui_style": get_ui_component_style("mixed")
            }
        ]
    
    return groups


def generate_meaningful_label(functions: List[Dict], user_name: str = "") -> str:
    """기능들의 특성을 분석해서 의미있는 라벨 생성"""
    
    if len(functions) == 1:
        # 단일 기능인 경우 제목 사용
        func = functions[0]
        if func.get("title"):
            return func["title"]
        return "추천 기능"
    
    # 기능들의 제목/부제목에서 키워드 추출
    keywords = []
    for func in functions:
        title = func.get("title", "").lower()
        subtitle = func.get("subtitle", "").lower()
        
        # 의미있는 키워드 추출
        meaningful_keywords = extract_meaningful_keywords(title + " " + subtitle)
        keywords.extend(meaningful_keywords)
    
    # 키워드 빈도 분석
    keyword_counts = Counter(keywords)
    
    if keyword_counts:
        # 가장 빈도가 높은 키워드로 라벨 생성
        top_keyword = keyword_counts.most_common(1)[0][0]
        label = generate_keyword_based_label(top_keyword, functions)
        
        # 사용자 이름이 있으면 개인화
        if user_name:
            return f"{user_name}님의 {label}"
        return label
    
    # 키워드가 없으면 기능 개수 기반으로 라벨 생성
    if len(functions) <= 3:
        if user_name:
            return f"{user_name}님의 추천 기능"
        return "추천 기능"
    else:
        if user_name:
            return f"{user_name}님의 주요 기능"
        return "주요 기능"


def extract_meaningful_keywords(text: str) -> List[str]:
    """텍스트에서 의미있는 키워드 추출"""
    keywords = []
    
    # 서비스 관련 키워드
    service_keywords = [
        "추천", "보안", "금융", "계좌", "결제", "송금", "대출", "투자",
        "쇼핑", "혜택", "할인", "포인트", "적립", "건강", "운동",
        "여행", "교통", "생활", "편의", "배달", "통신", "보험"
    ]
    
    for keyword in service_keywords:
        if keyword in text:
            keywords.append(keyword)
    
    # 특별한 패턴이 있는지 확인
    if "추천" in text or "맞춤" in text:
        keywords.append("추천")
    if "보안" in text or "인증" in text or "잠금" in text:
        keywords.append("보안")
    if "혜택" in text or "할인" in text or "%" in text:
        keywords.append("혜택")
    if "포인트" in text or "적립" in text:
        keywords.append("포인트")
    
    return keywords


def generate_keyword_based_label(keyword: str, functions: List[Dict]) -> str:
    """키워드 기반으로 자연스러운 라벨 생성"""
    
    # 키워드별 자연스러운 라벨 매핑
    keyword_labels = {
        "추천": "추천 서비스",
        "보안": "보안 서비스", 
        "금융": "금융 서비스",
        "계좌": "계좌 관리",
        "결제": "결제 서비스",
        "송금": "송금 서비스",
        "대출": "대출 서비스",
        "투자": "투자 서비스",
        "쇼핑": "쇼핑 서비스",
        "혜택": "특별 혜택",
        "할인": "할인 정보",
        "포인트": "포인트 서비스",
        "적립": "적립 서비스",
        "건강": "건강 관리",
        "운동": "운동 서비스",
        "여행": "여행 서비스",
        "교통": "교통 서비스",
        "생활": "생활 서비스",
        "편의": "편의 서비스",
        "배달": "배달 서비스",
        "통신": "통신 서비스",
        "보험": "보험 서비스"
    }
    
    base_label = keyword_labels.get(keyword, f"{keyword} 서비스")
    
    # 기능 개수에 따른 조정
    if len(functions) <= 3:
        return base_label
    else:
        return f"{base_label} 모음"


def get_ui_component_style(component_type: str) -> Dict[str, str]:
    """UI 컴포넌트 타입별 스타일 정보"""
    
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
