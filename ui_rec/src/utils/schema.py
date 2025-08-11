from typing import List, Dict, Any


def ui_response_template(user_id: str, layout_density: str, functions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "home": {
            "layout_density": layout_density,  # low / medium / high
            "functions": functions,  # list of {function_id, include, component_type, service_cluster, label, order}
        }
    } 