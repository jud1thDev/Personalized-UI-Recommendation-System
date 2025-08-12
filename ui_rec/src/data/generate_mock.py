import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.io import read_yaml, ensure_dir, write_csv_with_timestamp
from ..utils.time import access_time_cluster


CFG_DATA = read_yaml("ui_rec/config/data.yaml")
CFG_MODEL = read_yaml("ui_rec/config/model.yaml")
RAW_DIR = CFG_DATA["paths"]["raw_dir"]

random.seed(42)
np.random.seed(42)


def generate_users(n_users: int = 5000) -> pd.DataFrame:
    age_groups = ["10s","20s","30s","40s","50s","60s","70s+"]
    devices = ["Android","iOS"]
    users = []
    for i in range(n_users):
        uid = f"U{i:05d}"
        ag = np.random.choice(age_groups, p=[0.08,0.22,0.24,0.2,0.14,0.09,0.03])
        is_senior = ag in ["60s","70s+"]
        device = np.random.choice(devices, p=[0.65,0.35])
        users.append((uid, ag, is_senior, device))
    return pd.DataFrame(users, columns=["user_id","age_group","is_senior","device_type"])


# KB 스타뱅킹 앱 기준
def generate_functions(n_functions: int = 20) -> pd.DataFrame:
    functions = [
        # 계좌/자산 관련
        ("f001", "account", "KB마이핏통장", "429102-04-129287"),
        ("f002", "account", "KB스타 건강적금", "건강하고 든든한 적금"),
        ("f003", "account", "KB스타 투자상품", "리스크에 맞는 포트폴리오"),
        
        # 금융 서비스
        ("f004", "finance", "KB금융그룹 계열사 상품까지 한번에!", "내게 맞는 대출 찾기"),
        ("f005", "finance", "신용카드 혜택 한눈에", "포인트 적립 현황"),
        ("f006", "finance", "보험 상품 추천", "나에게 맞는 보험 찾기"),
        
        # 생활 서비스
        ("f007", "lifestyle", "홈 화면 계좌와 알림을 바로 확인", "빠른 로그인 설정하기"),
        ("f008", "lifestyle", "일정 관리 및 알림", "스마트 캘린더 연동"),
        ("f009", "lifestyle", "개인정보 설정", "보안 및 개인정보 관리"),
        
        # 건강/포인트
        ("f010", "health", "오늘 2,416 걸음", "건강 적금 연동"),
        ("f011", "health", "용돈 받기 매일 랜덤", "포인트 적립"),
        ("f012", "health", "식물 키우 포인트", "건강한 습관 만들기"),
        
        # 쇼핑/혜택
        ("f013", "shopping", "맞춤 쇼핑 추천", "KB카드 할인 혜택"),
        ("f014", "shopping", "할인 정보 알림", "관심 상품 가격 변동"),
        ("f015", "shopping", "배송 현황 추적", "실시간 배송 위치 확인"),
        
        # 여행/교통
        ("f016", "travel", "교통 정보 실시간", "대중교통 및 교통 상황"),
        ("f017", "travel", "여행 일정 관리", "항공권 및 숙박 예약"),
        ("f018", "travel", "내비게이션 서비스", "실시간 경로 안내"),
        
        # 추천 서비스
        ("f019", "recommendation", "7,500원에 10GB 든든하게!", "가입도 간단해서 걱정 없어요"),
        ("f020", "recommendation", "맞춤 콘텐츠 추천", "관심사 기반 개인화")
    ]
    
    return pd.DataFrame(functions, columns=["function_id","service_cluster","title","subtitle"])


def simulate_events(users: pd.DataFrame, funcs: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    component_types = ["card", "list_item", "banner", "grid_item"]
    
    rows = []
    start = datetime.now() - timedelta(days=days)
    for _, u in tqdm(users.iterrows(), total=len(users)):
        user_base_sessions = np.random.poisson(lam=2.5) + 1  # 일평균 세션 수
        for d in range(days):
            day = start + timedelta(days=d)
            n_sessions = max(0, int(np.random.normal(user_base_sessions, 1.0)))
            for s in range(n_sessions):
                session_id = f"{u.user_id}-{day.strftime('%Y%m%d')}-{s:02d}"
                n_events = np.random.randint(2, 10)
                current_funcs = funcs.sample(np.random.randint(3, min(10, len(funcs))))
                for _ in range(n_events):
                    f = current_funcs.sample(1).iloc[0]
                    ts = day + timedelta(minutes=int(np.random.uniform(0, 24*60)))
                    dwell = max(1, np.random.exponential(scale=60))  # 초 단위 체류
                    clicked = np.random.rand() < 0.35
                    component_type = np.random.choice(component_types)
                    component_id = f"cmp-{f.function_id}-{np.random.randint(1000):04d}"
                    position = np.random.randint(1, 50)
                    rows.append([
                        u.user_id, u.age_group, u.is_senior, u.device_type,
                        f.function_id, f.service_cluster, f.title, f.subtitle,
                        session_id, ts, access_time_cluster(ts),
                        dwell, int(clicked), component_id, position, component_type
                    ])
    cols = [
        "user_id","age_group","is_senior","device_type",
        "function_id","service_cluster","title","subtitle",
        "session_id","timestamp","access_time_cluster",
        "dwell_seconds","clicked","component_id","component_position","component_type"
    ]
    df = pd.DataFrame(rows, columns=cols)
    return df


def main():
    ensure_dir(RAW_DIR)
    users = generate_users(n_users=20)  # 개발용: 20명으로 제한
    funcs = generate_functions()
    events = simulate_events(users, funcs, days=7)  # 7일로 제한
    path = write_csv_with_timestamp(events, base_name="events", out_dir=RAW_DIR)
    print(f"Saved events to {path}")


if __name__ == "__main__":
    main() 