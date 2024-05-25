from flask import Flask, jsonify, request

from ragnarok import retrieve_and_generate

app = Flask(__name__)

@app.route('/api/model/<string:model_path>/collection/<string:dataset>/retriever/<string:host>/reranker/<string:host_reranker>/query=<string:query>&hits_retriever=<int:top_k_retrieve>&hits_reranker=<int:top_k_rerank>&qid=<int:qid>', methods=['GET'])
def search(model_path, dataset, host, host_reranker, query,top_k_retrieve, top_k_rerank,qid):
    try:
        # Assuming the function is called with these parameters and returns a response
        response = retrieve_and_generate.retrieve_and_generate(
            dataset=dataset,
            query=query,
            model_path=model_path,
            host="http://localhost:" + host_reranker,
            interactive=True, 
            k=[top_k_retrieve, top_k_rerank],
            qid=qid,
        )
        result = {
            "topic_id": response.query.qid,
            "topic": response.query.text,
            "references": response.references,
            "response_length": sum(len(sentence["text"]) for sentence in response.answer),
            "answer": [{"text": sentence["text"], "citations": sentence["citations"]} for sentence in response.answer]
        }
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# http://localhost:8082/ base url 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083, debug=True)
