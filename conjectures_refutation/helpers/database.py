import os
import sqlite3

DB_PATH = "db/counterexamples.db"

def check_if_counterexample_exists(g6: str) -> bool:
    connexion = sqlite3.connect(DB_PATH)
    cursor = connexion.cursor()
    requete = "SELECT 1 FROM graph WHERE g6_counterexample = ?"
    cursor.execute(requete, (g6,))

    resultat = cursor.fetchone()

    connexion.close()

    return resultat is not None

def check_if_db_exists() -> bool:
    return os.path.exists(DB_PATH)

def create_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    connexion = sqlite3.connect(DB_PATH)
    cursor = connexion.cursor()

    create_table = """
                   CREATE TABLE IF NOT EXISTS graph
                   (   id INTEGER PRIMARY KEY AUTOINCREMENT,
                       g6_counterexample TEXT NOT NULL UNIQUE
                   )
                   """

    cursor.execute(create_table)
    connexion.commit()
    connexion.close()

def insert_counterexample(g6: str):
    connexion = sqlite3.connect(DB_PATH)
    cursor = connexion.cursor()
    requete = "INSERT INTO graph (g6_counterexample) VALUES (?)"
    cursor.execute(requete, (g6,))

    connexion.commit()
    connexion.close()