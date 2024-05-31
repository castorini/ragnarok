#!/bin/bash

# Read the query from the terminal
read -p "Enter your query: " query

# Encode the query
#!/bin/bash

# Read the query from the terminal

# Encode the query using sed
encoded_query=$(echo "$query" | sed -e 's/ /%20/g')

# Make the curl request
# Make the curl request
curl -X GET "http://localhost:8084/api/model/command-r-plus/index/msmarco-v2.1-doc-segmented/8081/reranker/rank_zephyr/8082/query=$encoded_query&hits_retriever=100&hits_reranker=20&qid=1"
curl -X GET "http://localhost:8084/api/model/command-r-plus/index/msmarco-v2.1-doc-segmented/8081/reranker/gpt_4o/8083/query=$encoded_query&hits_retriever=100&hits_reranker=20&qid=1"
