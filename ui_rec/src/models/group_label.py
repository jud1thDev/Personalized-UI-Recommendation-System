from .common import load_latest_features, train_multiclass


def main():
    df = load_latest_features()
    path = train_multiclass(df, target="service_cluster_label", model_key="group_label", model_name="lgbm_group_label")
    print(f"Saved group_label model to {path}")


if __name__ == "__main__":
    main() 