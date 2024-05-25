from datetime import datetime
from pathlib import Path
from typing import List

from tqdm import tqdm

from ragnarok.generate.llm import LLM
from ragnarok.data import Request, Result, DataWriter


class RAG:
    def __init__(self, agent: LLM) -> None:
        self._agent = agent

    def answer_batch(
        self,
        requests: List[Request],
        topk: int = 20,
        shuffle_candidates: bool = False,
        logging: bool = False,
    ) -> List[Result]:
        """
        Generates a list of attributed answers using the Ragnarok agent.

        Args:
            requests (List[Request]): The list of requests. Each request has a query and a candidates list.
            topk (int, optional): The end rank for processing. Defaults to 20.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before answering. Defaults to False.
            logging (bool, optional): Enables logging of the answering process. Defaults to False.

        Returns:
            List[Result]: A list containing the attributed answers.
        """
        results = []
        for request in tqdm(requests):
            result = self._agent.answer_batch(
                [request],
                topk=min(topk, len(request.candidates)),
                shuffle_candidates=shuffle_candidates,
                logging=logging,
            )
            results.append(result[0])
        return results

    def answer(
        self,
        request: Request,
        rank_start: int = 0,
        rank_end: int = 20,
        shuffle_candidates: bool = False,
        logging: bool = False,
    ) -> Result:
        """
        Generates an attributed answer using the Ragnarok agent.

        Args:
            request (Request): The answering request which has a query and a candidates list.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 20.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before answering. Defaults to False.
            logging (bool, optional): Enables logging of the answering process. Defaults to False.

        Returns:
            Result: the generated result which contains the attributed answer.
        """
        results = self.answer_batch(
            requests=[request],
            rank_start=rank_start,
            rank_end=rank_end,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
        )
        return results[0]

    def write_answer_results(
        self,
        retrieval_method_name: str,
        results: List[Result],
        shuffle_candidates: bool = False,
        top_k_candidates: int = 20,
        dataset_name: str = None,
        results_dirname: str = "results",
    ) -> str:
        """
        Writes the attributed answers to files in specified formats.

        This function saves the results only in the JSON format expected in TREC 2024 RAG.

        Args:
            retrieval_method_name (str): The name of the retrieval method.
            results (List[Result]): The results to be written.
            shuffle_candidates (bool, optional): Indicates if the candidates were shuffled. Defaults to False.
            top_k_candidates (int, optional): The number of top candidates considered. Defaults to 20.
            dataset_name (str, optional): The name of the dataset used. Defaults to None.
            results_dirname (str, optional): The directory name to save the results. Defaults to "results".

        Returns:
            str: The file name of the saved results.

        Note:
            The function creates directories and files as needed. The file names are constructed based on the
            provided parameters and the current timestamp to ensure uniqueness so there are no collisions.
        """
        _modelname = self._agent._model.split("/")[-1]
        if _modelname.startswith("checkpoint"):
            _modelname = self._agent._model.split("/")[-2] + "_" + _modelname
        name = f"{_modelname}_{self._agent._context_size}_{top_k_candidates}_{self._agent._prompt_mode}"
        if dataset_name:
            name = f"{name}_{dataset_name}"
        if self._agent._num_few_shot_examples > 0:
            name += f"_{self._agent._num_few_shot_examples}_shot"
        name = (
            f"{name}_shuffled_{datetime.isoformat(datetime.now())}"
            if shuffle_candidates
            else f"{name}_{datetime.isoformat(datetime.now())}"
        )
        # write generate results
        writer = DataWriter(results)
        Path(f"{results_dirname}/{retrieval_method_name}/").mkdir(
            parents=True, exist_ok=True
        )
        writer.write_in_json_format(
            f"{results_dirname}/{retrieval_method_name}/{name}.json"
        )
        return f"{results_dirname}/{retrieval_method_name}/{name}.json"
