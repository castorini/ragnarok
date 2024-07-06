import sqlite3
from typing import Tuple, List
from enum import Enum 

BASE=10
SCALE=400
INIT_RATING=1000
K=5

# need to run gradio web_server.py from ./src/ragnarok/api/ directory 
#sql-lite connection 
conn = sqlite3.connect('elo.db')
cursor = conn.cursor()

class BattleResult(Enum):
    answer_a = "answer_a" 
    answer_tie = "answer_tie"
    answer_b = "answer_b"
    evidence_a = "evidence_a"
    evidence_tie = "evidence_tie"
    evidence_b = "evidence_b"

    def is_answer(self):
        return self.name.startswith("answer_")
    def is_evidence(self):
        return self.name.startswith("evidence_")
    def get_score(self):
    # return the score that A got (1 for win, 0 for loss, 0.5 for tie)
        table = {
            BattleResult.answer_a: 1,
            BattleResult.answer_tie: 0.5,
            BattleResult.answer_b: 0,
            BattleResult.evidence_a: 1,
            BattleResult.evidence_tie: 0.5,
            BattleResult.evidence_b: 0.5,
        }
        return table[self]

class BattleInfo:
    def __init__(self, llm_a: str, llm_b: str, retriever_a: str, retriever_b: str, reranker_a: str, reranker_b: str):
        self.llm_a = llm_a
        self.llm_b = llm_b
        self.retriever_a = retriever_a
        self.retriever_b = retriever_b
        self.reranker_a = reranker_a
        self.reranker_b = reranker_b

class Leaderboards(Enum):
    llm="llm"
    retrieve="retrieve"
    rag="rag"

class EloType(Enum):
    answer="answer_elo"
    evidence="evidence_elo"

# Implementation of the Elo rating system: https://en.wikipedia.org/wiki/Elo_rating_system
def compute_elo(sa: int, ra: int=INIT_RATING, rb: int=INIT_RATING):
    ea = 1 / (1 + BASE ** ((rb-ra)/400))
    eb = 1 / (1 + BASE ** ((ra-rb)/400))

    ra_new = ra + K*(sa - ea)
    rb_new = rb + K*(1-sa-eb)

    return (ra_new,rb_new)


def get_score(name: str, leaderboard: str, elo_type: str):
    conn = sqlite3.connect('elo.db')
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT {elo_type} FROM {leaderboard} WHERE name = ?", (name,))
        rating = cursor.fetchone()
        if rating:
            return rating[0]
        else:
            cursor.execute(f"INSERT INTO {leaderboard} (name, {elo_type}) VALUES (?,?)", (name, INIT_RATING))
            conn.commit()
            return get_score(name, leaderboard, elo_type)
    
    except sqlite3.Error as e:
        return f"An error has occured while fetching the score for {name}: {e}"
    finally:
        conn.close()
    
    
def get_score_pair(names: Tuple[str, str], leaderboard: str, elo_type: str):
    return (get_score(names[0],leaderboard,elo_type), get_score(names[1],leaderboard, elo_type))
    

def insert_score(name: str, new_elo: int, leaderboard: str, elo_type: str): 
    conn = sqlite3.connect('elo.db')
    cursor = conn.cursor()
    try: 
        cursor.execute(f"""
            UPDATE {leaderboard}
            SET {elo_type} = ?
            WHERE name = ?
        """, (new_elo, name))
        conn.commit()
        # print(f"updated {name},{elo_type} in {leaderboard} to {new_elo}")
    except sqlite3.Error as e:
        return f"An error has occured while pushing the score for {name}: {e}"
    finally:
        conn.close()
    
def insert_score_pair(names: Tuple[str,str], new_elos: Tuple[int,int], leaderboard:str, elo_type: str):
    return (insert_score(names[0], new_elos[0], leaderboard, elo_type), insert_score(names[1], new_elos[1], leaderboard, elo_type))



def get_leaderboards(info: BattleInfo):
    valid_leaderboards = []
    if info.llm_a==info.llm_b:
        valid_leaderboards.append(Leaderboards.retrieve)
    elif (info.retriever_a, info.reranker_b)==(info.retriever_a, info.reranker_a):
        valid_leaderboards.append(Leaderboards.llm)
    
    # always a valid entry into rag leaderboard 
    valid_leaderboards.append(Leaderboards.rag)
    return valid_leaderboards

def handle_battle(result: BattleResult, info: BattleInfo):
    if (info.llm_a, info.reranker_a, info.retriever_a)==(info.llm_b, info.reranker_b, info.retriever_b):
        return () # don't change elo since pipelines are the same
    
    valid_leaderboards: List[Leaderboards] = get_leaderboards(info)
    if Leaderboards.retrieve in valid_leaderboards and result.is_answer():
        return () # not allowed to evaluate answer elo for retriever leaderboard
    elo_type = EloType.answer if result.is_answer() else EloType.evidence
    for leaderboard in valid_leaderboards:
        entry_names = ("","")
        if leaderboard==Leaderboards.llm:
            entry_names=(info.llm_a, info.llm_b)
        elif leaderboard==Leaderboards.retrieve:
            entry_names=(f"{info.retriever_a}+{info.reranker_a}",f"{info.retriever_b}+{info.reranker_b}")
        else:
            entry_names=(f"{info.retriever_a}+{info.reranker_a}+{info.llm_a}",f"{info.retriever_b}+{info.reranker_b}+{info.llm_b}")
        (ra,rb) = get_score_pair(entry_names, leaderboard.value, elo_type.value)
        (ra_new, rb_new) = compute_elo(result.get_score(), int(ra),int(rb))
        insert_score_pair(entry_names, (ra_new,rb_new), leaderboard.value, elo_type.value)

    return ()