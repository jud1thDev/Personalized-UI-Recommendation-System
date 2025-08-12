from .common import load_latest_features, train_multiclass


def main():
    df = load_latest_features()
    path = train_multiclass(df, target="ui_type_label", model_key="ui_type", model_name="ui_type")
    print(f"Saved ui_type model to {path}")


if __name__ == "__main__":
    main()
