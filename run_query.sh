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
curl -X GET "http://localhost:8083/api/model/command-r-plus/collection/msmarco-v2.1-doc-segmented/retriever/8081/reranker/8082/query=$encoded_query&hits_retriever=40&hits_reranker=40&qid=1"
