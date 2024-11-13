import argparse

from flask import Flask, jsonify

from ragnarok import retrieve_and_generate
from ragnarok.generate.os_llm import OSLLM
from ragnarok.retrieve_and_rerank.retriever import RetrievalMethod


def create_app(os_model_path=None):
    app = Flask(__name__)

    # Load the OSLLM model if os_model_path is provided
    osllm_model = OSLLM(os_model_path) if os_model_path else None

    if os_model_path:
        print(f"Loaded OSLLM model from {os_model_path}")

    @app.route(
        "/api/model/<string:model_path>/index/<string:dataset>/<string:host>/reranker/<string:reranker_method>/<string:host_reranker>/query=<string:query>&hits_retriever=<int:top_k_retrieve>&hits_reranker=<int:top_k_rerank>&qid=<int:qid>",
        methods=["GET"],
    )
    def search(
        model_path,
        dataset,
        host,
        reranker_method,
        host_reranker,
        query,
        top_k_retrieve,
        top_k_rerank,
        qid,
    ):
        try:
            if reranker_method == "rank_zephyr":
                retrieval_method = [RetrievalMethod.BM25, RetrievalMethod.RANK_ZEPHYR]
            elif reranker_method == "gpt_4o":
                retrieval_method = [RetrievalMethod.BM25, RetrievalMethod.RANK_GPT4O]
            else:
                return jsonify({"error": "Invalid reranker method"}), 400

            # Assuming the function is called with these parameters and returns a response
            response = retrieve_and_generate.retrieve_and_generate(
                dataset=dataset,
                retrieval_method=retrieval_method,
                query=query,
                model_path=model_path,
                host="http://localhost:" + host_reranker,
                interactive=True,
                k=[top_k_retrieve, top_k_rerank],
                qid=qid,
                osllm_model=osllm_model,  # Pass the loaded OSLLM model
            )
            result = {
                "topic_id": response.query.qid,
                "topic": response.query.text,
                "references": response.references,
                "response_length": sum(
                    len(sentence["text"]) for sentence in response.answer
                ),
                "answer": [
                    {"text": sentence["text"], "citations": sentence["citations"]}
                    for sentence in response.answer
                ],
            }
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def main():
    parser = argparse.ArgumentParser(description="Start the Ragnarok Flask server.")
    parser.add_argument(
        "--os_model_path",
        type=str,
        help="Path to the OSLLM model. Can be None.",
    )
    parser.add_argument(
        "--port", type=int, default=8084, help="The port to run the Flask server on."
    )

    args = parser.parse_args()
    app = create_app(args.os_model_path)
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
