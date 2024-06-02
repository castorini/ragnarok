import requests
from urllib import parse
from typing import Any, Dict, List, Union

from ragnarok.data import Request, Candidate, Query
from ragnarok.retrieve_and_rerank.retriever import RetrievalMode, RetrievalMethod


class Restriever:
    def __init__(
        self,
        retrieval_mode: RetrievalMode = RetrievalMode.DATASET,
        retrieval_method: List[RetrievalMethod] = [RetrievalMethod.BM25, RetrievalMethod.RANK_ZEPHYR],
    ) -> None:
        """
        Creates a ServiceRetriever instance with a specified retrieval method and mode.

        Args:
            retrieval_mode (RetrievalMode): The retrieval mode to be used. Defaults to DATASET. Only DATASET mode is currently supported.
            retrieval_method (List[RetrievalMethod]): The retrieval method(s) to be used. Defaults to BM25 followed by RankZephyr. Only this is currently supported.

        Raises:
            ValueError: If retrieval mode or retrieval method is invalid or missing.
        """
        self._retrieval_mode = retrieval_mode
        self._retrieval_method = retrieval_method

        if retrieval_mode != RetrievalMode.DATASET:
            raise ValueError(f"{retrieval_mode} is not supported for ServiceRetriever. Only DATASET mode is currently supported.")
        if retrieval_method[0] != RetrievalMethod.BM25 and retrieval_method[1] != RetrievalMethod.RANK_ZEPHYR:
            raise ValueError(f"{retrieval_method} is not supported for ServiceRetriever. Only BM25 + RankZephyr is currently supported.")
        if not retrieval_method:
            raise "Please provide a retrieval method."
        if retrieval_method == RetrievalMethod.UNSPECIFIED:
            raise ValueError(f"Invalid retrieval method: {retrieval_method}. Please provide a specific retrieval method.")

    @staticmethod
    def from_dataset_with_prebuilt_index(
        dataset_name: str,
        host_reranker: str,
        host_retriever: str,
        retriever_path: str = "bm25",
        reranker_path: str = "rank_zephyr",
        k: List[int] = [100, 100],
        request: Request = None,
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
        
        try:
            retriever_path = RetrievalMethod.from_string(retriever_path.lower())
            reranker_path = RetrievalMethod.from_string(reranker_path.lower())
        except KeyError:
            retriever_path = RetrievalMethod.UNSPECIFIED
            reranker_path = RetrievalMethod.UNSPECIFIED
        
        retrieval_method = [retriever_path, reranker_path]

        if RetrievalMethod.UNSPECIFIED in retrieval_method:
            raise ValueError(
                f"Invalid retrieval method: {retrieval_method}. Please provide a specific retrieval method."
            )
        retriever = Restriever(
            RetrievalMode.DATASET,
            retrieval_method=retrieval_method,
        )
        return retriever.retrieve(dataset=dataset_name, k=k, host_retriever=host_retriever, host_reranker=host_reranker, request=request)

    def retrieve(
        self,
        dataset: Union[str, List[str], List[Dict[str, Any]]],
        request: Request,
        host_reranker: str = "8082",
        host_retriever: str = "8081",
        k: List[int] = [100, 20],
    ) -> Request:
        """
        Executes the retrieval process based on the configation provided with the Retriever instance. Takes in a Request object with a query and empty candidates object and the top k items to retrieve. 

        Args:
            request (Request): The request containing the query and qid. 
            dataset (str): The name of the dataset.
            k (int, optional): The top k hits to retrieve. Defaults to 100.
            host (str): The Anserini API host address. Defaults to http://localhost:8082

        Returns:
            Request. Contains a query and list of candidates
        Raises:
            ValueError: If the retrieval mode is invalid or the result format is not as expected.
        """

        parsed_query = parse.quote(request.query.text)
        rerank_method = self._retrieval_method[-1]
        retrieval_method = self._retrieval_method[0]

        url = f"http://localhost:{host_reranker}/api/model/{rerank_method}/index/{dataset}/{host_retriever}?query={parsed_query}&hits_retriever={str(k[0])}&hits_reranker={str(k[1])}&qid={request.query.qid}&retrieval_method={retrieval_method}"
        print(url)
        response = requests.get(url)
        print(response)
        if response.ok:
            data = response.json()
            print(data)
            retrieved_results = Request(
                query = Query(text = data["query"]["text"], qid = data["query"]["qid"])
            )
            candidates = []
            for candidate in data["candidates"]:
                candidates.append(Candidate(
                    docid = candidate["docid"],
                    score = candidate["score"],
                    doc = candidate["doc"],
                ))
            retrieved_results.candidates = candidates
        else: 
            raise ValueError(f"Failed to retrieve data from RankLLM server. Error code: {response.status_code}")
        return retrieved_results