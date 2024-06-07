"""
Trace nugget trajectory


Usage:
    python3 src/ragnarok/evaluate/trace_nugget_trajectory.py <input_file>
"""

import json
import argparse
from difflib import ndiff

def diff_color_text(old, new):
    diff = list(ndiff(old, new))
    diff_text = []
    for i in diff:
        if i.startswith("+ "):
            diff_text.append(f"\033[92m{i[2:]};\033[0m")  # Green for additions
        elif i.startswith("- "):
            diff_text.append(f"\033[91m{i[2:]};\033[0m")  # Red for deletions
        elif i.startswith("  "):
            diff_text.append(f"{i[2:]};")  # No color for unchanged
    return " ".join(diff_text)

def main(input_file):
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            print(data["qid"], data["text"])
            print()

            # iterate through the nugget_trajectory to capture differences across iterations
            nugget_trajectory = data["nugget_trajectory"]

            previous_nuggets = [""]
            diff_outputs = []

            for i, current_nuggets in enumerate(nugget_trajectory):
                if previous_nuggets:
                    diff_outputs.append(diff_color_text(previous_nuggets, current_nuggets))
                else:
                    diff_outputs.append(" ".join(current_nuggets))  # First iteration, no previous to compare
                previous_nuggets = current_nuggets

            # Render with color-coded differences
            for i, diff_output in enumerate(diff_outputs):
                print(f"iteration {i+1}:")
                print(diff_output)
                print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a JSONL file to display nugget trajectories with color-coded differences.')
    parser.add_argument('input_file', type=str, help='The input JSONL file')

    args = parser.parse_args()
    main(args.input_file)