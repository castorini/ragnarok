from typing import List
from urllib import parse

import requests

from ragnarok.data import Candidate, Query, Request
from ragnarok.retrieve_and_rerank.retriever import RetrievalMethod, RetrievalMode


class Restriever:
    def __init__(
        self,
        retrieval_mode: RetrievalMode = RetrievalMode.DATASET,
        retrieval_method: List[RetrievalMethod] = [
            RetrievalMethod.BM25,
            RetrievalMethod.RANK_ZEPHYR,
        ],
    ) -> None:
        """
        Creates a Restriever instance with a specified retrieval method and mode.

        Args:
            retrieval_mode (RetrievalMode): The retrieval mode to be used. Defaults to DATASET. Only DATASET mode is currently supported.
            retrieval_method (List[RetrievalMethod]): A list of length 2 that contains: [retrieval method, reranking method]. Defaults to BM25 followed by RankZephyr.

        Raises:
            ValueError: If retrieval mode or retrieval method is invalid or missing.
        """
        self._retrieval_mode = retrieval_mode
        self._retrieval_method = retrieval_method

        if retrieval_mode != RetrievalMode.DATASET:
            raise ValueError(
                f"{retrieval_mode} is not supported for ServiceRetriever. Only DATASET mode is currently supported."
            )
        if not retrieval_method:
            raise "Please provide a retrieval method."
        if retrieval_method == RetrievalMethod.UNSPECIFIED:
            raise ValueError(
                f"Invalid retrieval method: {retrieval_method}. Please provide a specific retrieval method."
            )

    @staticmethod
    def from_dataset_with_prebuilt_index(
        dataset_name: str,
        host_reranker: str,
        host_retriever: str,
        retriever_path: str = "bm25",
        reranker_path: str = "rank_zephyr",
        k: List[int] = [100, 100],
        request: Request = None,
        num_passes: int = 1,
    ):
        """
        Creates a Restriever instance for a dataset with a prebuilt index.

        Args:
            - dataset_name (str): The name of the dataset.
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

        # RetreivalMethod Options:
        # UNSPECIFIED = "unspecified"
        # BM25 = "bm25"
        # RANK_ZEPHYR = "rank_zephyr"
        # RANK_ZEPHYR_RHO = "rank_zephyr_rho"
        # RANK_VICUNA = "rank_vicuna"
        # RANK_GPT4O = "gpt-4o"
        # RANK_GPT4 = "gpt-4"
        # RANK_GPT35_TURBO = "gpt-3.5-turbo"

        try:
            retriever_path = RetrievalMethod.from_string(retriever_path.lower())
            reranker_path = RetrievalMethod.from_string(reranker_path.lower())
        except KeyError:
            retriever_path = RetrievalMethod.UNSPECIFIED
            reranker_path = RetrievalMethod.UNSPECIFIED

        retrieval_method = [retriever_path, reranker_path]

        # Rerank method can be none (RetrievalMethod.UNSPECIFIED)
        if retrieval_method[0] == RetrievalMethod.UNSPECIFIED:
            raise ValueError(
                f"Invalid retrieval method: {retrieval_method}. Please provide a specific retrieval method."
            )
        retriever = Restriever(
            RetrievalMode.DATASET,
            retrieval_method=retrieval_method,
        )
        return retriever.retrieve(
            dataset=dataset_name,
            k=k,
            host_retriever=host_retriever,
            host_reranker=host_reranker,
            request=request,
            num_passes=num_passes,
        )

    def retrieve(
        self,
        dataset: str,
        request: Request,
        host_reranker: str = "8082",
        host_retriever: str = "8081",
        k: List[int] = [20, 10],
        num_passes: int = 1,
    ) -> Request:
        """
        Executes the 2 stage retrieval+rerank process based on the configation provided with the Retriever instance. Takes in a Request object with a query and empty candidates object and the top k items to retrieve.

        Args:
            request (Request): The request containing the query and qid.
            dataset (str): The name of the dataset.
            k (List[int], optional): [top k hits to retrieve, top k hits to rerank]. Defaults to [20,10]
            host_reranker (str): The Anserini API host address. Defaults to 8081
            host_retriever (str): The Reranking API host address. Defaults to 8082
            num_passes (int): Number of passes to perform in the reranking stage

        Returns:
            Request. Contains a query and list of candidates
        Raises:
            ValueError: If the retrieval mode is invalid or the result format is not as expected.
        """

        parsed_query = parse.quote(request.query.text)
        (retrieval_method, rerank_method) = self._retrieval_method

        # API request URL specified in https://github.com/castorini/rank_llm api folder
        url = f"http://localhost:{host_reranker}/api/model/{rerank_method}/index/{dataset}/{host_retriever}?query={parsed_query}&hits_retriever={str(k[0])}&hits_reranker={str(k[1])}&qid={request.query.qid}&num_passes={num_passes}&retrieval_method={retrieval_method}"

        # Send request to Rerank API
        response = requests.get(url)

        # First 2 stages, retrieval and reranking, OK
        if response.ok:
            # Parse information about the query
            data = response.json()
            retrieved_results = Request(
                query=Query(text=data["query"]["text"], qid=data["query"]["qid"])
            )

            # Parse response into a list of candidates
            for candidate in data["candidates"]:
                retrieved_results.candidates.append(
                    Candidate(
                        docid=candidate["docid"],
                        score=candidate["score"],
                        doc=candidate["doc"],
                    )
                )

            print("Retrieval and reranking completed successfully...")
            print(
                f"Candidates provided during retrieval+reranking: {len(retrieved_results.candidates)}"
            )
        else:
            raise ValueError(
                f"Failed to retrieve data from RankLLM server. Error code: {response.status_code}"
            )
        return retrieved_results
