from .common import load_latest_features, train_binary


def main():
    df = load_latest_features()
    path = train_binary(df, target="exposure_label", model_name="exposure")
    print(f"Saved exposure model to {path}")


if __name__ == "__main__":
    main()
