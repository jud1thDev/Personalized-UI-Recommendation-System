import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

from ..utils.io import read_yaml, latest_file, write_csv_with_timestamp
from ..utils.io import ensure_dir
from ..utils.features import entropy

CFG_DATA = read_yaml("ui_rec/config/data.yaml")
RAW_DIR = CFG_DATA["paths"]["raw_dir"]
PROC_DIR = CFG_DATA["paths"]["processed_dir"]


def build_user_level(df: pd.DataFrame) -> pd.DataFrame:
    # 사용자 레벨 경로 엔트로피, 세션 수 등
    path_entropy = df.groupby(["user_id","session_id"])['function_id'].apply(list)
    user_entropy = path_entropy.groupby(level=0).apply(lambda s: np.mean([entropy(seq) for seq in s]))

    session_count = df.groupby(["user_id","session_id"]).size().groupby(level=0).size()
    fav_time = df.groupby("user_id")["access_time_cluster"].agg(lambda x: x.value_counts().index[0])

    user_df = pd.DataFrame({
        "tap_path_entropy": user_entropy,
        "session_count": session_count,
        "access_time_cluster": fav_time
    }).reset_index()
    return user_df


def build_user_function(df: pd.DataFrame) -> pd.DataFrame:
    # 사용자-기능 단위 집계
    grp = df.groupby(["user_id","function_id"])  
    entry_count = grp.size().rename("entry_count")
    click_rate = grp["clicked"].mean().rename("click_rate")
    visit_duration = grp["dwell_seconds"].mean().rename("visit_duration")

    # 재방문: 같은 기능 1시간 이내 재진입 횟수 근사
    df_sorted = df.sort_values(["user_id","function_id","timestamp"])  
    df_sorted["prev_ts"] = df_sorted.groupby(["user_id","function_id"])['timestamp'].shift(1)
    df_sorted["revisit"] = ((df_sorted["timestamp"] - df_sorted["prev_ts"]).dt.total_seconds() <= 3600).fillna(False)
    return_count = df_sorted.groupby(["user_id","function_id"])['revisit'].sum().rename("return_count")

    # 마지막 사용 시점 기준 일수
    max_ts = df["timestamp"].max()
    last_access_days = (max_ts - grp["timestamp"].max()).dt.days.rename("last_access_days")

    uf = pd.concat([entry_count, click_rate, visit_duration, return_count, last_access_days], axis=1).reset_index()
    return uf


def make_targets(uf: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    # 노출 여부 타깃: 클릭률/체류/재방문 가중 합이 임계치 이상
    score = (uf["click_rate"]*0.5 + (uf["visit_duration"]/uf["visit_duration"].max())*0.3 + (uf["return_count"]/max(1,uf["return_count"].max()))*0.2)
    uf["exposure_label"] = (score > score.quantile(0.5)).astype(int)

    # UI 유형 타깃: 과거 관찰된 component_type 최빈값
    ui_type = df.groupby(["user_id","function_id"])['component_type'].agg(lambda x: x.value_counts().index[0]).rename("ui_type_label")

    # 그룹/소제목 타깃: 기능 메타로부터(최빈 cluster/label)
    cluster = df.groupby(["user_id","function_id"])['service_cluster'].agg(lambda x: x.value_counts().index[0]).rename("service_cluster_label")
    label = df.groupby(["user_id","function_id"])['label'].agg(lambda x: x.value_counts().index[0]).rename("label_text")

    # 배치 순서 타깃: 사용자 내 기능별 점수 내림차순 순위(1 = 최상단)
    uf["rank_score"] = score
    uf["rank_label"] = uf.groupby("user_id")["rank_score"].rank(ascending=False, method="first")

    targets = uf.merge(ui_type.reset_index(), on=["user_id","function_id"], how="left") \
                .merge(cluster.reset_index(), on=["user_id","function_id"], how="left") \
                .merge(label.reset_index(), on=["user_id","function_id"], how="left")
    return targets


def main():
    events_path = latest_file(CFG_DATA["files"]["events_pattern"], RAW_DIR)
    if not events_path:
        raise FileNotFoundError("No events CSV found. Run generate_mock.py first.")
    df = pd.read_csv(events_path, parse_dates=["timestamp"])

    user_df = build_user_level(df)
    uf = build_user_function(df)

    # 사용자 레벨 피처 결합 + 메타 정보(나이/디바이스 등)
    meta = df.groupby("user_id")[ ["age_group", "is_senior", "device_type"] ].agg(lambda x: x.iloc[0]).reset_index()
    feat = uf.merge(user_df, on="user_id", how="left") 
    feat = feat.merge(meta, on="user_id", how="left")

    # 타깃 생성 및 병합
    targets = make_targets(uf, df)
    feat = feat.merge(targets.drop(columns=["entry_count","click_rate","visit_duration","return_count","last_access_days","rank_score"]),
                      on=["user_id","function_id"], how="left")

    path = write_csv_with_timestamp(feat, base_name="features", out_dir=PROC_DIR)
    print(f"Saved features to {path}")


if __name__ == "__main__":
    main() 