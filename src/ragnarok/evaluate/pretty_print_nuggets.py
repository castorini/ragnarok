import argparse
import json


def pretty_print_file(filename):
    dataset = []
    with open(filename, "r") as file:
        for line in file:
            data = json.loads(line)
            dataset.append(data)
    for data in dataset:
        qid = data["qid"].replace("_0", "")
        text = data["text"]
        nuggets = data["nuggets"]

        print(f"QID: {qid}\n")
        print(f"Query: {text}\n")
        print(f"# Nuggets: {len(nuggets)}\n")
        print("Nuggets:")
        for i, nugget in enumerate(nuggets, start=1):
            print(f" {i}) {nugget}")
        print("\n\n\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretty print JSON file content.")
    parser.add_argument(
        "filename", type=str, help="The filename of the JSON file to be pretty printed"
    )
    args = parser.parse_args()
    pretty_print_file(args.filename)
