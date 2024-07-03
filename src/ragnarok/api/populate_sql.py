import sqlite3
import pandas as pd 

INIT_RATING = 1000

conn = sqlite3.connect('elo.db')
cursor = conn.cursor()

retriever_options = ["bm25"]
reranker_options = ["rank_zephyr", "rank_vicuna", "gpt-4o", "unspecified"]
llm_options = ["command-r", "command-r-plus", "gpt-4o", "gpt-35-turbo", "gpt-4"]

cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS llm (
        name TEXT PRIMARY KEY,
        answer_elo INTEGER DEFAULT 1000,
        evidence_elo INTEGER DEFAULT 1000
    )
''')

cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS rag (
        name TEXT PRIMARY KEY,
        answer_elo INTEGER DEFAULT 1000,
        evidence_elo INTEGER DEFAULT 1000
    )
''')

cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS retrieve (
        name TEXT PRIMARY KEY,
        evidence_elo INTEGER DEFAULT 1000
    )
''')

conn.commit()  

# LLM generation 
def insert_llm(name: str):
    cursor.execute('SELECT name FROM llm WHERE name = ?', (name,))
    if cursor.fetchone() is None:
        cursor.execute('INSERT INTO llm (name) VALUES (?)', (name,))

# Retrieval pipeline: retrieval+rerank
def insert_retrieve(name: str):
    cursor.execute('SELECT name FROM retrieve WHERE name = ?', (name,))
    if cursor.fetchone() is None:
        cursor.execute('INSERT INTO retrieve (name) VALUES (?)', (name,))

# RAG pipeline: retrieval+rerank+generation
def insert_rag(name: str):
    cursor.execute('SELECT name FROM rag WHERE name = ?', (name,))
    if cursor.fetchone() is None:
        cursor.execute('INSERT INTO rag (name) VALUES (?)', (name,))

# Populate llm table
for llm in llm_options:
    insert_llm(llm)

# Populate retrieve table with retriever and reranker combinations
for retriever in retriever_options:
    for reranker in reranker_options:
        retrieve_name = f"{retriever}+{reranker}"
        insert_retrieve(retrieve_name)

# Populate rag table with combinations of retrieve+reranker+llm
for retriever in retriever_options:
    for reranker in reranker_options:
        for llm in llm_options:
            rag_name = f"{retriever}+{reranker}+{llm}"
            insert_rag(rag_name)

df_llm = pd.read_sql_query("SELECT * FROM llm", conn)
df_retrieve = pd.read_sql_query("SELECT * FROM retrieve", conn)
df_rag = pd.read_sql_query("SELECT * FROM rag", conn)


conn.commit() 
conn.close()  
