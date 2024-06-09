import json
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


class CacheInputFormat(Enum):
    JSON = "json"
    JSONL = "jsonl"

    def __str__(self):
        return self.value


class RetrievalMethod(Enum):
    UNSPECIFIED = "unspecified"
    BM25 = "bm25"
    RANK_ZEPHYR = "rank_zephyr"
    RANK_ZEPHYR_RHO = "rank_zephyr_rho"
    RANK_VICUNA = "rank_vicuna"
    RANK_GPT4O = "gpt-4o"
    RANK_GPT4 = "gpt-4"
    RANK_GPT35_TURBO = "gpt-3.5-turbo"

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(value):
        for method in RetrievalMethod:
            if method.value == value:
                return method
        raise ValueError(f"Unknown retrieval method: {value}")


class Retriever:
    def __init__(
        self,
        retrieval_mode: RetrievalMode,
        dataset: Union[str, List[str], List[Dict[str, Any]]],
        retrieval_method: Union[
            RetrievalMethod, List[RetrievalMethod]
        ] = RetrievalMethod.UNSPECIFIED,
        query: str = None,
        index_path: str = None,
        topics_path: str = None,
    ) -> None:
        self._retrieval_mode = retrieval_mode
        self._dataset = dataset
        self._retrieval_method = (
            retrieval_method
            if isinstance(retrieval_method, list)
            else [retrieval_method]
        )
        self._query = query
        self._index_path = index_path
        self._topics_path = topics_path

    @staticmethod
    def from_dataset_with_prebuilt_index(
        dataset_name: str,
        retrieval_method: Union[
            RetrievalMethod, List[RetrievalMethod]
        ] = RetrievalMethod.BM25,
        cache_input_format: CacheInputFormat = CacheInputFormat.JSONL,
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
        return retriever.retrieve(k=k, cache_input_format=cache_input_format)

    def retrieve(
        self,
        retrieve_results_dirname: str = "retrieve_results",
        cache_input_format: CacheInputFormat = CacheInputFormat.JSONL,
        k: List[int] = [100, 20],
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
                f"{retrieve_results_dirname}/{self._retrieval_method[-1].name}/retrieve_results_{self._dataset}_top{k[-1]}.{cache_input_format}"
            )
            print(f"Loading candidates from {candidates_file}.")
            if not candidates_file.is_file():
                # Get top100 file instead
                try:
                    candidates_file = Path(
                        f"{retrieve_results_dirname}/{self._retrieval_method[-1].name}/retrieve_results_{self._dataset}_top100.{cache_input_format}"
                    )
                    print()
                    if not candidates_file.is_file():
                        raise ValueError(f"File not found: {candidates_file}")
                except ValueError as e:
                    print(f"Failed to load JSON file: {candidates_file}")
            if candidates_file.is_file():
                if cache_input_format == CacheInputFormat.JSON:
                    with open(candidates_file, "r") as f:
                        loaded_results = json.load(f)
                elif cache_input_format == CacheInputFormat.JSONL:
                    with open(candidates_file, "r") as f:
                        loaded_results = [json.loads(l) for l in f]
                retrieved_results = [
                    from_dict(data_class=Request, data=r) for r in loaded_results
                ]
                # TODO remove!!! Filter those if query.qid has _ and does not end with 0
                retrieved_results = [
                    r
                    for r in retrieved_results
                    if "_" not in r.query.qid or r.query.qid.endswith("0")
                ]
                # Ensure the candidates are of at most k length
                for r in retrieved_results:
                    r.candidates = r.candidates[: k[-1]]
                print(
                    f"Loaded {len(retrieved_results)} requests from {candidates_file}."
                )
        else:
            raise ValueError(f"Invalid retrieval mode: {self._retrieval_mode}")
        return retrieved_results
