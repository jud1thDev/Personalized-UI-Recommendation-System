import argparse
import subprocess


def run(cmd: list[str]):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["generate-mock","build-features","train-all","predict"]) 
    p.add_argument("--allowed", type=str, default="card,list_item,banner,icon")
    args = p.parse_args()

    if args.command == "generate-mock":
        run(["python", "-m", "ui_rec.src.data.generate_mock"])
    elif args.command == "build-features":
        run(["python", "-m", "ui_rec.src.features.build_features"])
    elif args.command == "train-all":
        run(["python", "-m", "ui_rec.src.models.exposure"])
        run(["python", "-m", "ui_rec.src.models.ui_type"])
        run(["python", "-m", "ui_rec.src.models.group_label"])
        run(["python", "-m", "ui_rec.src.models.rank"])
    elif args.command == "predict":
        run(["python", "-m", "ui_rec.src.inference.predict", "--allowed", args.allowed])


if __name__ == "__main__":
    main() 