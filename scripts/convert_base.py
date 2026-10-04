"""Convert local Cosmos understanding and vision tensors into a Qwen3-VL base."""
import argparse
import json

from nswm.assets import convert_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard-mib", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(convert_base(args.source, args.processor, args.out, args.shard_mib), indent=2))


if __name__ == "__main__":
    main()
