from typing import List, Dict, Any


def ui_response_template(user_id: str, functions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    UI 응답 템플릿
    
    컴포넌트 타입:
    - 일반 사용자:
        * list_single: 나의 총자산 (1개 항목)
        * list_multi: 이번 주 카드결제, 오늘한 지출 (2~5개 항목)
        * grid_multi: 오늘 걸음, 용돈 받기 (2~4개 항목)
        * grid_full: 청년 혜택들 (1개 항목)
        * banner: 추천 서비스 (3~4개 항목)
    
    - 시니어 사용자:
        * s_list_single: 단일 리스트 (1개 항목)
        * s_list_multi: 다중 리스트 (2~5개 항목)
        * s_grid_multi: 다중 그리드 (2개 항목)
        * s_grid_full: 전체 그리드 (1개 항목)
    
    - 공통:
        * exchange_stock_widget: 환율·증시 위젯
    """
    return {
        "user_id": user_id,
        "home": {
            "functions": functions,  # list of {function_id, include, component_type, service_cluster, label, order}
        }
    }


def get_component_types_by_user_type(is_senior: bool) -> List[str]:
    """사용자 타입에 따른 사용 가능한 컴포넌트 타입 반환"""
    if is_senior:
        return [
            "s_list_single",
            "s_list_multi", 
            "s_grid_multi",
            "s_grid_full"
        ]
    else:
        return [
            "list_single",
            "list_multi",
            "grid_multi", 
            "grid_full",
            "banner"
        ]


def get_exchange_stock_component_type() -> str:
    """환율·증시 전용 컴포넌트 타입 반환"""
    return "exchange_stock_widget" 