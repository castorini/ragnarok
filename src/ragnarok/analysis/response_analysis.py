import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple, Union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from ragnarok.data import Result


class ResponseAnalyzer:
    def __init__(
        self,
        data: Union[List[str], List[Result]],
    ) -> None:
        self._data = data

    @staticmethod
    def from_inline_results(results: List[Result]) -> "ResponseAnalyzer":
        """
        Method to create a ResponseAnalyzer instance from a list of Result objects.

        Args:
            results (List[Result]): A list of Result objects.

        Returns:
            ResponseAnalyzer: An instance of the ResponseAnalyzer.
        """
        return ResponseAnalyzer(data=results)

    @staticmethod
    def from_stored_files(filenames: List[str]) -> "ResponseAnalyzer":
        """
        Method to create to create a ResponseAnalyzer instance from a list of filenames.

        Args:
            filenames (List[str]): A list of filenames where each file contains data to be analyzed.

        Returns:
            ResponseAnalyzer: An instance of the ResponseAnalyzer.
        """
        data = []
        for filename in filenames:
            with open(filename, "r") as file:
                file_data = json.load(file)
                data.extend(file_data)
        return ResponseAnalyzer(data=data)

    def read_results_responses(self) -> Tuple[List[str], List[int]]:
        """
        Reads responses from the specified list of Result objects and produces the total number of passages.

        Returns:
            Tuple[List[str], List[int]]: A tuple object containing a list of responses and a list of corresponding numbers of passages.
        """
        num_passages = []
        responses = []
        for result in self._data:
            for exec_info in result.rag_exec_summary:
                responses.append(exec_info.response)
                num_passage = self._get_num_passages(exec_info.prompt)
                num_passages.append(int(num_passage))
        return responses, num_passages

    def read_saved_responses(self) -> Tuple[List[str], List[int]]:
        """
        Reads responses from the specified list of files and produces the total number of passages.

        Returns:
            Tuple[List[str], List[int]]: A tuple object containing a list of responses and a list of corresponding numbers of passages.
        """
        num_passages = []
        responses = []
        for result in self._data:
            with open(result) as f:
                rag_exec_summaries = json.load(f)
            for summary in rag_exec_summaries:
                for exec_info in summary.rag_exec_summary:
                    responses.append(exec_info.response)
                    num_passage = self._get_num_passages(exec_info.prompt)
                    num_passages.append(int(num_passage))
        return responses, num_passages

    def read_responses(self) -> Tuple[List[str], List[int]]:
        """
        Selects what read response class method to call depending on the input type.

        Returns:
            Tuple[List[str], List[int]]: A tuple object containing a list of responses and a list of corresponding numbers of passages.
        """
        if all(isinstance(item, str) for item in self._data):
            return self.read_saved_responses()
        elif all(isinstance(item, Result) for item in self._data):
            return self.read_results_responses()
        else:
            raise ValueError(
                "Input data must be a list of file paths or a list of Result objects."
            )

    def count_errors(
        self, verbose: bool = False, normalize: bool = False
    ) -> Dict[str, Union[int, float]]:
        """
        Counts an array of different types of errors in the given responses.

        Args:
        verbose (bool, optional): When enabled, the analyzer will print out the malformed responses. Defaults to False.
        normalize (bool, optional): When enabled, the returned dictionary will be normalized. Defaults to False.

        Returns:
            Dict[str, Union[int, float]]: A dictionary object containing (normalized) counts of different types of errors.
        """
        responses, num_passages = self.read_responses()

        stats_dict = {
            "ok": 0,
            "wrong_format": 0,
        }
        for resp, num_passage in zip(responses, num_passages):
            stats_dict["ok"] += 1

        if not normalize:
            return stats_dict

        # Create normalized dicts
        normalized_stats_dict = {}
        for key in stats_dict:
            normalized_stats_dict[key] = (stats_dict[key] / len(responses)) * 100.0
            # Round to two decimal places
            normalized_stats_dict[key] = round(normalized_stats_dict[key], 2)
        return normalized_stats_dict


def main(args):
    if args.files:
        response_analyzer = ResponseAnalyzer.from_stored_files(args.files)
    else:
        print("Error: Please specify the files containing generation summaries.")
        sys.exit(1)

    error_counts = response_analyzer.count_errors(args.verbose)
    print("Normalized scores:", error_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", nargs="+", help="Filenames of generation summaries", required=False
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose output of errors"
    )
    args = parser.parse_args()

    main(args)
