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
    
    # 1단계: 기능을 서비스 클러스터와 순위 기반으로 그룹화
    initial_groups = create_initial_function_groups(functions)
    
    # 2단계: 각 그룹에 최적의 컴포넌트 타입 결정
    optimized_groups = assign_optimal_component_types(initial_groups, functions[0].get("is_senior", False))
    
    # AI 라벨 생성기를 한 번만 생성 (싱글톤 패턴으로 중복 방지)
    ai_generator = None
    try:
        from ..utils.ai_label_generator import create_ai_label_generator
        ai_generator = create_ai_label_generator()
    except Exception as e:
        print(f"AI label generator initialization failed: {e}")
        ai_generator = None
    
    # 3단계: AI 라벨 생성 및 최종 결과 생성
    return create_final_groups(optimized_groups, functions, ai_generator, user_name)


def create_initial_function_groups(functions: List[Dict]) -> List[Dict]:
    """기능을 서비스 클러스터와 순위 기반으로 초기 그룹화"""
    
    # 서비스 클러스터별로 그룹화
    cluster_groups = {}
    for func in functions:
        cluster = func.get("service_cluster", "unknown")
        if cluster not in cluster_groups:
            cluster_groups[cluster] = []
        cluster_groups[cluster].append(func)
    
    # 각 클러스터 그룹을 크기에 따라 세분화
    initial_groups = []
    for cluster, funcs in cluster_groups.items():
        # 순위별로 정렬
        funcs.sort(key=lambda x: x.get("rank", 999))
        
        # 크기에 따라 그룹 분할
        if len(funcs) <= 3:
            # 작은 그룹은 그대로 유지
            initial_groups.append({
                "functions": funcs,
                "service_cluster": cluster,
                "size": len(funcs)
            })
        else:
            # 큰 그룹은 적절한 크기로 분할
            max_group_size = 5
            for i in range(0, len(funcs), max_group_size):
                group_funcs = funcs[i:i + max_group_size]
                initial_groups.append({
                    "functions": group_funcs,
                    "service_cluster": cluster,
                    "size": len(group_funcs)
                })
    
    return initial_groups


def assign_optimal_component_types(groups: List[Dict], is_senior: bool) -> List[Dict]:
    """각 그룹에 최적의 컴포넌트 타입 할당"""
    
    for group in groups:
        size = group["size"]
        
        if is_senior:
            # 시니어 사용자: 단순하고 큰 컴포넌트 선호
            if size == 1:
                group["component_type"] = "s_list_single"
            elif size <= 3:
                group["component_type"] = "s_list_multi"
            elif size <= 5:
                group["component_type"] = "s_grid_multi"
            else:
                group["component_type"] = "banner"
        else:
            # 일반 사용자: 다양하고 세밀한 컴포넌트 선호
            if size == 1:
                group["component_type"] = "list_single"
            elif size <= 3:
                group["component_type"] = "list_multi"
            elif size <= 5:
                group["component_type"] = "grid_multi"
            else:
                group["component_type"] = "banner"
    
    return groups


def create_final_groups(optimized_groups: List[Dict], all_functions: List[Dict], ai_generator, user_name: str) -> List[Dict]:
    """최종 그룹 생성 및 AI 라벨 생성"""
    
    final_groups = []
    
    for group in optimized_groups:
        funcs = group["functions"]
        comp_type = group["component_type"]
        
        # AI 기반 라벨 생성 (3개 이상 기능)
        if len(funcs) >= 3 and ai_generator:
            try:
                # 서비스 클러스터 정보 추출
                service_cluster = funcs[0].get("service_cluster", "unknown")
                label = ai_generator.generate_label(funcs, service_cluster)
            except Exception as e:
                print(f"AI label generation failed for {comp_type}: {e}")
                label = generate_smart_label(funcs, comp_type, user_name)
        else:
            # 1-2개 기능은 스마트 라벨 생성
            label = generate_smart_label(funcs, comp_type, user_name)
        
        # JSON 표시 시 중복 데이터 제거 (user_id, is_senior)
        clean_funcs = []
        for func in funcs:
            clean_func = {k: v for k, v in func.items() 
                         if k not in ['user_id', 'is_senior']}
            clean_funcs.append(clean_func)
        
        final_groups.append({
            "label": label,
            "functions": clean_funcs,
            "component_type": comp_type,
            "function_count": len(funcs),
            "ui_style": get_ui_component_style(comp_type)
        })
    
    return final_groups


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


def generate_smart_label(functions: List[Dict], component_type: str, user_name: str = "") -> str:
    """컴포넌트 타입과 기능 특성을 고려한 스마트한 라벨 생성"""
    
    # 서비스 클러스터 분석
    cluster_counts = Counter()
    for func in functions:
        cluster = func.get('service_cluster', 'unknown')
        cluster_counts[cluster] += 1
    
    # 주요 서비스 클러스터 파악
    if cluster_counts:
        main_cluster = cluster_counts.most_common(1)[0][0]
        
        # 컴포넌트 타입별로 적절한 라벨 생성
        if "single" in component_type:
            # single: 중요한 단일 기능
            return get_single_component_label(functions[0], user_name)
        
        elif "banner" in component_type:
            # banner: 홍보/이벤트성 기능
            return get_banner_label(main_cluster, user_name)
        
        elif "grid" in component_type:
            # grid: 시각적 표현이 중요한 기능
            return get_grid_label(main_cluster, user_name)
        
        else:
            # list: 일반적인 기능
            return get_list_label(main_cluster, user_name)
    
    return f"{user_name}님의 추천 기능" if user_name else "추천 기능"


def get_single_component_label(func: Dict, user_name: str = "") -> str:
    """single 컴포넌트용 라벨 (중요한 단일 기능)"""
    
    # 기능의 구체적인 특성에 따른 라벨 생성
    service_cluster = func.get("service_cluster", "unknown")
    function_id = func.get("function_id", "")
    
    # 서비스 클러스터별 구체적인 라벨
    if service_cluster == "finance":
        if "transfer" in function_id.lower() or "송금" in function_id:
            return "계좌이체"
        elif "payment" in function_id.lower() or "결제" in function_id:
            return "결제하기"
        elif "loan" in function_id.lower() or "대출" in function_id:
            return "대출신청"
        elif "investment" in function_id.lower() or "투자" in function_id:
            return "투자상품"
        else:
            return "금융서비스"
    
    elif service_cluster == "account":
        if "login" in function_id.lower() or "로그인" in function_id:
            return "로그인"
        elif "profile" in function_id.lower() or "프로필" in function_id:
            return "프로필"
        elif "security" in function_id.lower() or "보안" in function_id:
            return "보안설정"
        else:
            return "계정관리"
    
    elif service_cluster == "lifestyle":
        if "utility" in function_id.lower() or "공과금" in function_id:
            return "공과금납부"
        elif "mobile" in function_id.lower() or "휴대폰" in function_id:
            return "휴대폰결제"
        elif "transport" in function_id.lower() or "교통" in function_id:
            return "교통카드"
        else:
            return "생활서비스"
    
    elif service_cluster == "health":
        if "medical" in function_id.lower() or "의료" in function_id:
            return "의료서비스"
        elif "fitness" in function_id.lower() or "운동" in function_id:
            return "운동관리"
        else:
            return "건강관리"
    
    elif service_cluster == "shopping":
        if "coupon" in function_id.lower() or "쿠폰" in function_id:
            return "쿠폰서비스"
        elif "delivery" in function_id.lower() or "배송" in function_id:
            return "배송조회"
        else:
            return "쇼핑서비스"
    
    elif service_cluster == "travel":
        if "hotel" in function_id.lower() or "호텔" in function_id:
            return "호텔예약"
        elif "flight" in function_id.lower() or "항공" in function_id:
            return "항공예약"
        else:
            return "여행서비스"
    
    else:
        # 기본적으로 서비스 클러스터 기반 라벨
        return get_simple_fallback_label(service_cluster, user_name)


def get_banner_label(cluster: str, user_name: str = "") -> str:
    """banner 컴포넌트용 라벨 (홍보/이벤트성)"""
    
    banner_labels = {
        "혜택": "특별 혜택",
        "이벤트": "진행 중인 이벤트",
        "공지": "중요 공지사항",
        "lifestyle": "생활 혜택",
        "shopping": "쇼핑 혜택",
        "travel": "여행 혜택",
        "finance": "금융 혜택"
    }
    
    base_label = banner_labels.get(cluster, f"{cluster} 혜택")
    return f"{user_name}님의 {base_label}" if user_name else base_label


def get_grid_label(cluster: str, user_name: str = "") -> str:
    """grid 컴포넌트용 라벨 (시각적 표현이 중요한 기능)"""
    
    grid_labels = {
        "자산관리": "자산 현황",
        "건강": "건강 관리",
        "생활": "생활 서비스",
        "finance": "금융 현황",
        "account": "계정 정보",
        "shopping": "쇼핑 정보",
        "travel": "여행 정보"
    }
    
    base_label = grid_labels.get(cluster, f"{cluster} 정보")
    return f"{user_name}님의 {base_label}" if user_name else base_label


def get_list_label(cluster: str, user_name: str = "") -> str:
    """list 컴포넌트용 라벨 (일반적인 기능)"""
    
    list_labels = {
        "account": "계정 관리",
        "finance": "금융 서비스",
        "lifestyle": "생활 서비스",
        "health": "건강 관리",
        "shopping": "쇼핑 서비스",
        "travel": "여행 서비스",
        "security": "보안 설정"
    }
    
    base_label = list_labels.get(cluster, f"{cluster} 서비스")
    return f"{user_name}님의 {base_label}" if user_name else base_label


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
