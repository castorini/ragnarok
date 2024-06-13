import argparse
import json
from pprint import pprint

def pretty_print_file(filename):
    dataset = []
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append(data)
    for data in dataset:
        qid = data['query']['qid'].replace("_0", "")
        text = data['query']['text']
        candidates = data['candidates']
        example_label = {}
        for candidate in candidates:
            if candidate["judgment"] not in example_label:
                example_label[candidate["judgment"]] = candidate['doc']["segment"]
        print(f"QID: {qid}\n")
        print(f"Query: {text}\n")
        for ex in range(4):
            if ex not in example_label:
                print(f"Label: {ex} None")
                continue
            print(f"Label:{ex} {example_label[ex]}")
        print("\n\n\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretty print JSON file content.')
    parser.add_argument('filename', type=str, help='The filename of the JSON file to be pretty printed')
    args = parser.parse_args()
    pretty_print_file(args.filename)