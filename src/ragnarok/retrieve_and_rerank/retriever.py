import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

from dacite import from_dict

from ragnarok.data import Request


class RetrievalMode(Enum):
    DATASET = "dataset"
    CUSTOM = "custom"

    def __str__(self):
        return self.value


class RetrievalMethod(Enum):
    UNSPECIFIED = "unspecified"
    BM25 = "bm25"
    RANK_ZEPHYR = "rank_zephyr"

    def __str__(self):
        return self.value


class Retriever:
    def __init__(
        self,
        retrieval_mode: RetrievalMode,
        dataset: Union[str, List[str], List[Dict[str, Any]]],
        retrieval_method: Union[RetrievalMethod, List[RetrievalMethod]] = RetrievalMethod.UNSPECIFIED,
        query: str = None,
        index_path: str = None,
        topics_path: str = None,
    ) -> None:
        self._retrieval_mode = retrieval_mode
        self._dataset = dataset
        self._retrieval_method = retrieval_method if isinstance(retrieval_method, list) else [retrieval_method]
        self._query = query
        self._index_path = index_path
        self._topics_path = topics_path

    @staticmethod
    def from_dataset_with_prebuilt_index(
        dataset_name: str,
        retrieval_method: Union[RetrievalMethod, List[RetrievalMethod]] = RetrievalMethod.BM25,
        k: List[int] = [100, 20],
    ):
        """
        Creates a Retriever instance for a dataset with a prebuilt index.

        Args:
            dataset_name (str): The name of the dataset.
            retrieval_method (Union[RetrievalMethod, List[RetrievalMethod]]): The retrieval method(s) to be used. Defaults to BM25.
            k (List[int], optional): The top k hits to retrieve. Defaults to [100, 20].

        Returns:
            List[Request]: The list of requests. Each request has a query and list of candidates.

        Raises:
            ValueError: If dataset name or retrieval method is invalid or missing.
        """
        if not dataset_name:
            raise ValueError("Please provide name of the dataset.")
        if not isinstance(dataset_name, str):
            raise ValueError(
                f"Invalid dataset format: {dataset_name}. Expected a string representing name of the dataset."
            )
        if not retrieval_method:
            raise "Please provide a retrieval method."
        if RetrievalMethod.UNSPECIFIED in retrieval_method:
            raise ValueError(
                f"Invalid retrieval method: {retrieval_method}. Please provide a specific retrieval method."
            )
        retriever = Retriever(
            RetrievalMode.DATASET,
            dataset=dataset_name,
            retrieval_method=retrieval_method,
        )
        return retriever.retrieve(k=k)

    def retrieve(
        self, retrieve_results_dirname: str = "retrieve_results", k: List[int] = [100, 20]
    ) -> List[Request]:
        """
        Executes the retrieval process based on the configuration provided with the Retriever instance.

        Returns:
            List[Request]: The list of requests. Each request has a query and list of candidates.

        Raises:
            ValueError: If the retrieval mode is invalid or the format is not as expected.
        """
        if self._retrieval_mode == RetrievalMode.DATASET:
            candidates_file = Path(
                f"{retrieve_results_dirname}/{self._retrieval_method[-1].name}/retrieve_results_{self._dataset}_top{k[-1]}.jsonl"
            )
            if not candidates_file.is_file():
                # Read JSON file
                try:
                    # Replace jsonl with json in path
                    candidates_file = str(candidates_file)
                    with open(candidates_file.replace("jsonl", "json"), "r") as f:
                        loaded_results = json.load(f)
                    retrieved_results = [
                        from_dict(data_class=Request, data=r) for r in loaded_results
                    ]
                except ValueError as e:
                    print(f"Failed to load JSON file: {candidates_file}")
            else:
                # Load JSONL
                with open(candidates_file, "r") as f:
                    loaded_results = [json.loads(l) for l in f]
                retrieved_results = [
                    from_dict(data_class=Request, data=r) for r in loaded_results
                ]
        else:
            raise ValueError(f"Invalid retrieval mode: {self._retrieval_mode}")
        return retrieved_results
