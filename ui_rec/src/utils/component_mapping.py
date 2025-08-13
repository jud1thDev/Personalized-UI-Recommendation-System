"""
사용자 타입에 따른 UI 컴포넌트 타입 매핑
"""

def get_component_type_by_user_type(ui_type_pred, is_senior):
    """
    사용자 타입에 따라 컴포넌트 타입 결정
    
    Args:
        ui_type_pred (int): 예측된 UI 타입 (0-4)
        is_senior (bool): 시니어 사용자 여부
        
    Returns:
        str: 사용자 타입에 맞는 컴포넌트 타입
    """
    if is_senior:
        # 시니어 사용자용 컴포넌트 타입
        senior_mapping = {
            0: "s_list_single",
            1: "s_list_multi", 
            2: "s_grid_multi",
            3: "s_grid_full",
            4: "s_list_single"  # 기본값
        }
        return senior_mapping.get(ui_type_pred, "s_list_single")
    else:
        # 일반 사용자용 컴포넌트 타입
        normal_mapping = {
            0: "list_single",
            1: "list_multi",
            2: "grid_multi", 
            3: "grid_full",
            4: "banner"
        }
        return normal_mapping.get(ui_type_pred, "list_single")
