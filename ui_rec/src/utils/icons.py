"""
아이콘 제안 유틸리티 모듈
서비스 클러스터에 따른 적절한 아이콘 제안
"""

from typing import Dict


def get_icon_suggestion(service_cluster: str) -> Dict[str, str]:
    """서비스 클러스터에 따른 추천 아이콘 제안"""
    icon_mapping = {
        "account": {
            "icon_name": "account_balance",
            "icon_type": "material",  # Material Design Icons
            "icon_color": "#1976d2",
            "description": "계좌/자산 아이콘",
            "icon_size": "24px",
            "icon_url": "https://fonts.gstatic.com/s/i/materialicons/account_balance/v1/24px.svg" # 아이콘 링크는 임의로 넣어놓음
        },
        "finance": {
            "icon_name": "account_balance_wallet",
            "icon_type": "material",
            "icon_color": "#4caf50",
            "description": "지갑/금융 아이콘",
            "icon_size": "24px",
            "icon_url": "https://fonts.gstatic.com/s/i/materialicons/account_balance_wallet/v1/24px.svg"
        },
        "lifestyle": {
            "icon_name": "home",
            "icon_type": "material",
            "icon_color": "#9c27b0",
            "description": "홈/생활 서비스 아이콘",
            "icon_size": "24px",
            "icon_url": "https://fonts.gstatic.com/s/i/materialicons/home/v1/24px.svg"
        },
        "health": {
            "icon_name": "favorite",
            "icon_type": "material",
            "icon_color": "#f44336",
            "description": "하트/건강 아이콘",
            "icon_size": "24px",
            "icon_url": "https://fonts.gstatic.com/s/i/materialicons/favorite/v1/24px.svg"
        },
        "shopping": {
            "icon_name": "shopping_cart",
            "icon_type": "material",
            "icon_color": "#ff9800",
            "description": "쇼핑카트 아이콘",
            "icon_size": "24px",
            "icon_url": "https://fonts.gstatic.com/s/i/materialicons/shopping_cart/v1/24px.svg"
        },
        "travel": {
            "icon_name": "flight",
            "icon_type": "material",
            "icon_color": "#2196f3",
            "description": "비행기/여행 아이콘",
            "icon_size": "24px",
            "icon_url": "https://fonts.gstatic.com/s/i/materialicons/flight/v1/24px.svg"
        },
        "recommendation": {
            "icon_name": "star",
            "icon_type": "material",
            "icon_color": "#ffc107",
            "description": "별/추천 아이콘",
            "icon_size": "24px",
            "icon_url": "https://fonts.gstatic.com/s/i/materialicons/star/v1/24px.svg"
        }
    }
    
    return icon_mapping.get(service_cluster, {
        "icon_name": "star",
        "icon_type": "material",
        "icon_color": "#9c27b0",
        "description": "기본 별 아이콘",
        "icon_size": "24px",
        "icon_url": "https://fonts.gstatic.com/s/i/materialicons/star/v1/24px.svg"
    })
