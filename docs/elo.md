## Elo Leaderboard feature

Uses an implementation of the Elo rating system: https://en.wikipedia.org/wiki/Elo_rating_system

## We have 3 leaderboards:
1. LLM:  generation LLM's (e.g.,`GPT-4o`, `command-r`)
2. Retrieve: the Retrieve pipeline - retriever model + reranker model (e.g., `BM25`+`rank_zephyr`)
3. RAG: the whole RAG pipeline - retriever model + reranker model + LLM (e.g., `BM25`+`rank_zephyr`+`GPT-4o`)

**A battle is eligible for updating a leaderboard following the conditions stated below:**
- LLM — all parameters in pipelines are the same *except* for `llm`
- Retrieve — all parameters in pipelines are the same *except* for `retriever and/or reranker`
- RAG — the `llm`'s are not equal *and* either the `retriever` or `reranker`'s are not equal 

