from datetime import datetime
from typing import Literal


def access_time_cluster(ts: datetime) -> Literal["dawn","morning","afternoon","evening","night"]:
    h = ts.hour
    if 5 <= h < 9:
        return "morning"
    if 9 <= h < 13:
        return "afternoon"
    if 13 <= h < 18:
        return "evening"
    if 18 <= h < 23:
        return "night"
    return "dawn" 