import argparse
import json
import numpy as np

def score_assignment_file(filename, query_level=False):
    dataset = []
    with open(filename, "r") as file:
        for line in file:
            data = json.loads(line)
            dataset.append(data)

    results = []

    total_vital_support_hits = 0
    total_vital_nuggets = 0
    total_okay_support_hits = 0
    total_okay_nuggets = 0
    total_weighted_score = 0

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
            continue

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
            
            vital_score = vital_support_hits / (vital_nuggets + 0.00001)
            okay_score = okay_support_hits / (okay_nuggets + 0.00001)
            total_score = (vital_support_hits + okay_support_hits) / (vital_nuggets + okay_nuggets + 0.00001)
            weighted_score = (2 * vital_support_hits + okay_support_hits) / (2 * vital_nuggets + okay_nuggets + 0.00001)

            results.append({
                "qid": qid,
                "query": text,
                "vital_support_hits": vital_support_hits,
                "vital_nuggets": vital_nuggets,
                "vital_score": vital_score,
                "okay_support_hits": okay_support_hits,
                "okay_nuggets": okay_nuggets,
                "okay_score": okay_score,
                "total_score": total_score,
                "weighted_score": weighted_score
            })

            total_vital_support_hits += vital_support_hits
            total_vital_nuggets += vital_nuggets
            total_okay_support_hits += okay_support_hits
            total_okay_nuggets += okay_nuggets
            total_weighted_score += weighted_score

    if query_level:
        # Print or save the results
        for result in results:
            print(f"QID: {result['qid']}")
            print(f"Query: {result['query']}")
            print(f"Vital Support Hits: {result['vital_support_hits']}/{result['vital_nuggets']} = {result['vital_score']:.2f}")
            print(f"Okay Support Hits: {result['okay_support_hits']}/{result['okay_nuggets']} = {result['okay_score']:.2f}")
            print(f"Total Support Hits: {result['vital_support_hits'] + result['okay_support_hits']}/{result['vital_nuggets'] + result['okay_nuggets']} = {result['total_score']:.2f}")
            print(f"Weighted Score: {result['weighted_score']:.2f}")
            print("\n")

    # Calculate and print aggregate scores
    aggregate_vital_score = np.mean([result['vital_score'] for result in results])
    aggregate_okay_score = np.mean([result['okay_score'] for result in results])
    aggregate_total_score = np.mean([result['total_score'] for result in results])
    aggregate_weighted_score = np.mean([result['weighted_score'] for result in results])

    print("Aggregate Scores:")
    print(f"Aggregate Vital Score: {aggregate_vital_score:.2f}")
    print(f"Aggregate Okay Score: {aggregate_okay_score:.2f}")
    print(f"Aggregate Total Score: {aggregate_total_score:.2f}")
    print(f"Aggregate Weighted Score: {aggregate_weighted_score:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate and assign a score based on vital/okay hits for each file.")
    parser.add_argument(
        "filename", type=str, help="The filename of the JSON file to be processed"
    )
    parser.add_argument(
        "--query_level", action="store_true", help="If True, print query-level scores"
    )
    args = parser.parse_args()
    score_assignment_file(args.filename, args.query_level)