import argparse
import json


def pretty_print_file(filename):
    dataset = []
    with open(filename, "r") as file:
        for line in file:
            data = json.loads(line)
            dataset.append(data)
    for data in dataset:
        qid = data['qid'].replace("_0", "")
        text = data['text']
        nuggets = data['nuggets']
        if 'nugget_labels' in data:
            nugget_labels = data['nugget_labels']
        if 'nugget_assignment' in data:
            nugget_assignment = data['nugget_assignment']
        if 'nugget_assignment' in data:
            nuggets = list(zip(nugget_labels, nuggets, nugget_assignment))
        elif 'nugget_labels' in data:
            nuggets = list(zip(nugget_labels, nuggets)) 
        else:
            pass

        print(f"QID: {qid}\n")
        print(f"Query: {text}\n")
        if 'answer_text' in data:
            print(f"Answer: {data['answer_text']}\n")
        print(f"# Nuggets: {len(nuggets)}\n")
        print("Nuggets:")
        for i, nugget in enumerate(nuggets, start=1):
            print(f" {i}) {nugget}")
        if 'nugget_assignment' in data:
            # Calculate vital support hits and okay support hits
            vital_support_hits = 0
            okay_support_hits = 0
            vital_nuggets = 0
            okay_nuggets = 0
            for nugget in nuggets:
                if nugget[0] == "vital":
                    vital_nuggets += 1
                    if nugget[2] == "support":
                        vital_support_hits += 1
                elif nugget[0] == "okay":
                    okay_nuggets += 1
                    if nugget[2] == "support":
                        okay_support_hits += 1
            print(f"\nVital Support Hits: {vital_support_hits}/{vital_nuggets} = {vital_support_hits/(vital_nuggets + 0.00001):.2f}")
            print(f"\nOkay Support Hits: {okay_support_hits}/{okay_nuggets} = {okay_support_hits/(okay_nuggets + 0.00001):.2f}")
            print(f"\nTotal Support Hits: {vital_support_hits + okay_support_hits}/{vital_nuggets + okay_nuggets} = {(vital_support_hits + okay_support_hits)/(vital_nuggets + okay_nuggets + 0.00001):.2f}")
        print("\n\n\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretty print JSON file content.")
    parser.add_argument(
        "filename", type=str, help="The filename of the JSON file to be pretty printed"
    )
    args = parser.parse_args()
    pretty_print_file(args.filename)
