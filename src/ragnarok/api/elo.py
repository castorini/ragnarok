import sqlite3
import ragnarok.api.web_server
from ragnarok.api.web_server import BattleInfo, BattleResult, Leaderboards, EloType
from typing import Tuple, List

BASE=10
SCALE=400
INIT_RATING=1000
K=5

conn = ragnarok.api.web_server.conn
cursor = conn.cursor()

# Implementation of the Elo rating system: https://en.wikipedia.org/wiki/Elo_rating_system
def compute_elo(sa: int, ra: int=INIT_RATING, rb: int=INIT_RATING):
    ea = 1 / (1 + BASE ** ((rb-ra)/400))
    eb = 1 / (1 + BASE ** ((ra-rb)/400))

    ra_new = ra + K*(sa - ea)
    rb_new = rb + K*(1-sa-eb)

    return (ra_new,rb_new)


def get_score(name: str, leaderboard: str, elo_type: str):
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
    
def get_score_pair(names: Tuple[str, str], leaderboard: str, elo_type: str):
    return (get_score(names[0],leaderboard,elo_type), get_score(names[1],leaderboard, elo_type))
    

def insert_score(name: str, score: int, leaderboard: str, elo_type: str): 
    try: 
        cursor.execute(f"INSERT INTO {leaderboard} (name, {elo_type}) VALUES (?,?)", (name, score))
        conn.commit()
    except sqlite3.Error as e:
        return f"An error has occured while pushing the score for {name}: {e}"
    
def insert_score_pair(names: Tuple[str,str], leaderboard:str, elo_type: str):
    return (insert_score(names[0], leaderboard, elo_type), insert_score(names[1], leaderboard, elo_type))



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
        (ra_new, rb_new) = compute_elo(result.get_score(), ra,rb)
        insert_score_pair(entry_names, leaderboard, elo_type)

    return ()