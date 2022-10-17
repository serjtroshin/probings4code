import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import sleep


def process_invalid(args, paths, name):
    path2invalids = defaultdict(list)
    for path in paths:
        with path.open("r") as f:
            try:
                path2invalids[path.parent].append(set(json.load(f)))
            except Exception as e:
                print(f"ERROR {e} loading: {str(path)}")
                sleep(2)

    for path in Path(f"{args.data_dir}").glob("*/*"):
        all_invalid = set()
        for model in path.glob("*"):
            for emb in model.glob("*"):
                for ids in path2invalids[emb]:
                    all_invalid.update(ids)
        print("all_invalid", path, len(all_invalid))
        with (path / name).open("w") as fout:
            json.dump(list(all_invalid), fout)


def main(args):
    # collect all invalid ids for each task
    # they will not be used for each model
    invalid_ids = []

    for path in Path(f"{args.data_dir}/").glob("*/*/*/*"):
        if len(list(path.glob("invalid_ids.json"))) < 1:
            print(f"Path: {path}")
        else:
            for train in path.glob("invalid_ids.json"):
                invalid_ids.append(train)

    process_invalid(args, invalid_ids, "all_invalid_ids.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("collect")
    parser.add_argument("--data_dir", type=str, default="CodeAnalysis")
    args = parser.parse_args()

    main(args)
