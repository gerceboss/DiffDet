import yaml
from train import train
from eval import evaluate

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"], help="train or eval")
    args = parser.parse_args()

    config = load_config()

    if args.mode == "train":
        train(config)
    elif args.mode == "eval":
        evaluate(config)
