from .common import load_latest_features, train_multiclass


def main():
    df = load_latest_features()
    path = train_multiclass(df, target="service_cluster_label", model_key="service_cluster", model_name="lgbm_service_cluster")
    print(f"Saved service_cluster model to {path}")


if __name__ == "__main__":
    main() 