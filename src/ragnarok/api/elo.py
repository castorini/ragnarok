import sqlite3
import ragnarok.api.web_server

BASE=10
SCALE=400
INIT_RATING=1000
K=5

# Implementation of the Elo rating system: https://en.wikipedia.org/wiki/Elo_rating_system
def compute_elo(sa: int, ra: int=INIT_RATING, rb: int=INIT_RATING):
    ea = 1 / (1 + BASE ** ((rb-ra)/400))
    eb = 1 / (1 + BASE ** ((ra-rb)/400))

    ra_new = ra + K*(sa - ea)
    rb_new = rb + K*(1-sa-eb)

    return (ra_new,rb_new)


def get_score(model_name: str):
    try:
        cursor.execute('SELECT score FROM elo WHERE model_name = ?', (model_name,))
        rating = cursor.fetchone()
        if rating:
            return rating[0]
        else:
            cursor.execute('INSERT INTO elo (model_name, score) VALUES (?,?)', (model_name, INIT_RATING))
            return get_score(model_name)
    
    except sqlite3.Error as e:
        return f"An error has occured while fetching the score for {model_name}: {e}"
    

def insert_score(model_name: str, score: int): 
    try: 
        cursor.execute('INSERT INTO elo (model_name, score) VALUES (?,?)', (model_name, score))
    except sqlite3.Error as e:
        return f"An error has occured while pushing the score for {model_name}: {e}"
        

def handle_battle(sa: int, model_a: str, model_b: str):
    ra = get_score(model_a)
    rb = get_score(model_b)
    (ra_new,rb_new) = compute_elo(sa,ra,rb)
    insert_score(model_a, ra_new)
    insert_score(model_b,rb_new)



