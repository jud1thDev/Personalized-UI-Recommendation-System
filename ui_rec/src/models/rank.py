from .common import load_latest_features, train_regression


def main():
    df = load_latest_features()
    path = train_regression(df, target="rank_label", model_name="rank")
    print(f"Saved rank model to {path}")


if __name__ == "__main__":
    main()
