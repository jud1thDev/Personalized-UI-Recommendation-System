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
def generate_functions(n_functions: int = 50) -> pd.DataFrame:
    functions = [
        # 가입상품관리
        ("f001", "가입상품관리", "예금 상품 가입", "안전하게 목돈을 불리는 예금 가입"),
        ("f002", "가입상품관리", "적금 상품 가입", "목표 금액까지 꾸준히 모으는 적금 가입"),
        ("f003", "가입상품관리", "펀드 가입 / 해지", "다양한 펀드 상품 가입 및 해지"),
        ("f004", "가입상품관리", "주식 계좌 개설", "주식 거래를 시작하는 첫걸음"),
        ("f005", "가입상품관리", "대출 조회", "보유 중인 대출 현황 확인"),
        ("f006", "가입상품관리", "대출 상환", "대출 원리금 상환 처리"),
        ("f007", "가입상품관리", "대출 신청", "필요 자금을 위한 대출 신청"),
        ("f008", "가입상품관리", "보험 조회", "가입한 보험 내역과 보장 확인"),

        # 자산관리
        ("f009", "자산관리", "자산 리포트 조회", "나의 전체 자산 현황 한눈에 보기"),
        ("f010", "자산관리", "소비 리포트 조회", "한 달 소비 패턴 분석"),
        ("f011", "자산관리", "일정 금액 자동 저축", "매달 자동으로 저축하기"),
        ("f012", "자산관리", "자산 목표 설정", "목표 금액과 기간 설정"),
        ("f013", "자산관리", "마이데이터 연결", "다른 금융기관 데이터 통합"),

        # 공과금
        ("f014", "공과금", "공과금 납부", "전기·수도·가스 요금 간편 납부"),
        ("f015", "공과금", "정부지원금 조회", "받을 수 있는 지원금 확인"),
        ("f016", "공과금", "세금 납부", "국세·지방세 납부하기"),

        # 외환
        ("f017", "외환", "외화 환전", "원하는 통화로 환전하기"),
        ("f018", "외환", "환율 계산기", "실시간 환율로 금액 계산"),
        ("f019", "외환", "해외 송금", "국외 계좌로 송금하기"),

        # 금융편의
        ("f020", "금융편의", "계좌 조회", "내 모든 계좌 잔액 확인"),
        ("f021", "금융편의", "송금 / 이체", "빠르고 안전한 송금"),
        ("f022", "금융편의", "자동이체 관리", "자동이체 등록·변경·해지"),
        ("f023", "금융편의", "입출금 알림 설정", "실시간 거래 알림 받기"),
        ("f024", "금융편의", "카드 사용내역 조회", "카드별 결제 내역 확인"),
        ("f025", "금융편의", "카드 결제일 조회", "다음 결제일과 금액 확인"),
        ("f026", "금융편의", "카드 이용한도 변경", "카드 사용 한도 조정"),
        ("f027", "금융편의", "카드 분실신고", "분실 시 카드 사용 중지"),
        ("f028", "금융편의", "간편인증 / 바이오 인증 설정", "지문·얼굴·PIN 등록"),
        ("f029", "금융편의", "친구에게 송금", "연락처로 간편 송금"),

        # 혜택
        ("f030", "혜택", "이벤트 / 혜택 확인", "진행 중인 이벤트와 할인 혜택"),
        ("f031", "혜택", "생활 혜택 추천", "맞춤형 생활 혜택 안내"),
        ("f032", "혜택", "멤버십", "KB멤버십 포인트 및 혜택"),

        # 생활/제휴
        ("f033", "생활/제휴", "지점 / ATM 위치 찾기", "가까운 지점·ATM 찾기"),
        ("f034", "생활/제휴", "챗봇 상담", "24시간 AI 상담 서비스"),
        ("f035", "생활/제휴", "고객센터 전화 연결", "상담원 연결하기"),

        # 테마별서비스
        ("f036", "테마별서비스", "KB금융그룹", "KB금융그룹 계열사 정보"),
        ("f037", "테마별서비스", "꿀팁정보", "금융 생활 유용한 정보"),
        ("f038", "테마별서비스", "청년정책/장학금 모아보기", "청년 지원 정책과 장학금 정보"),
        ("f039", "테마별서비스", "맘 편한 아이금융", "육아 가정을 위한 금융 서비스"),
        ("f040", "테마별서비스", "든든한 군TIP 생활관", "군인 생활 금융 정보"),
        ("f041", "테마별서비스", "참 편한 50 PLUS 라운지", "50대 이상 맞춤 서비스"),
        ("f042", "테마별서비스", "KB스타뱅킹 길라잡이", "앱 사용 가이드"),

        # 사장님+
        ("f043", "사장님+", "사장님 홈", "사업자 전용 메인 화면"),
        ("f044", "사장님+", "사업용 계좌조회", "사업자 계좌 잔액·내역"),
        ("f045", "사장님+", "사업자 금융상품", "사업자 전용 금융 상품 안내"),
        ("f046", "사장님+", "사장님을 위한 서비스", "창업·경영 지원 서비스"),
        ("f047", "사장님+", "사장님을 위한 뱅킹", "사업 운영 금융 관리")
    ]

    return pd.DataFrame(functions, columns=["function_id", "service_cluster", "title", "subtitle"])

def simulate_events(users: pd.DataFrame, funcs: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    # 일반 사용자 컴포넌트 타입
    component_types_normal = [
        "list_single",  # 국민은행 기존 홈 기준: 나의 총자산 // 1개
        "list_multi",   # 국민은행 기존 홈 기준: 이번 주 카드결제,오늘한 지출 // 2~5개
        "grid_multi",   # 국민은행 기존 홈 기준: 오늘 걸음, 용돈 받기 // 2~4개
        "grid_full",    # 국민은행 기존 홈 기준: 미리 챙겨주면 좋은 청년 혜택들 // 1개
        "banner"       # 국민은행 기존 홈 기준: 추천 서비스 // 3~4개
    ]

    # 시니어 사용자 컴포넌트 타입
    component_types_senior = [
        "s_list_single", # 국민은행 기존 홈 기준: 모든 게 다 이 컴포넌트 // 1개
        "s_list_multi", # 국민은행 기존 홈 기준: 모든 게 다 이 컴포넌트 // 2~5개
        "s_grid_multi", # 기존엔 없던 컴포넌트, 일반 사용자용 컴포넌트를 살짝 변형하여 정의 // 2개
        "s_grid_full", # 기존엔 없던 컴포넌트, 일반 사용자용 컴포넌트를 살짝 변형하여 정의 // 1개
    ]

    # 환율·증시 전용 컴포넌트
    component_type_exchange_stock = "exchange_stock_widget"
    
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
                    
                    # 사용자 타입에 따른 컴포넌트 타입 선택
                    if u.is_senior:
                        component_type = np.random.choice(component_types_senior)
                    else:
                        component_type = np.random.choice(component_types_normal)
                    
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