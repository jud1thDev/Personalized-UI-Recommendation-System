from typing import List, Dict, Any


def ui_response_template(user_id: str, functions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "home": {
            "functions": functions,  # list of {function_id, include, component_type, service_cluster, label, order}
        }
    } 