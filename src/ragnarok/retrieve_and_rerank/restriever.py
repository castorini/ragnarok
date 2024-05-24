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

    def retrieve(
        self,
        dataset: Union[str, List[str], List[Dict[str, Any]]],
        request: Request,
        host: str = "http://localhost:8082",
        retriever_port: str = "8081",
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
        url = f"{host}/api/model/rank_zephyr/collection/{dataset}/retriever/{retriever_port}/search?query={parsed_query}&hits_retriever={str(k[0])}&hits_reranker={str(k[1])}&qid={request.query.qid}"

        response = requests.get(url)
        if response.ok:
            data = response.json()
            retrieved_results = Request(
                query = Query(text = data["query"]["text"], qid = data["query"]["qid"])
            )
            candidates = []
            #TODO(ronak) - Replace 20 with k[1] when fixed in RankLLM
            for candidate in data["candidates"][:20]:
                candidates.append(Candidate(
                    docid = candidate["docid"],
                    score = candidate["score"],
                    doc = candidate["doc"],
                ))
            retrieved_results.candidates = candidates
        else: 
            raise ValueError(f"Failed to retrieve data from Anserini server. Error code: {response.status_code}")
        return retrieved_results