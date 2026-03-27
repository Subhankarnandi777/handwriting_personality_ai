import sqlite3
import os
from datetime import datetime
from src.utils.config import BASE_DIR

DB_PATH = os.path.join(BASE_DIR, "output", "history.db")

def _get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = _get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            image_name TEXT,
            openness REAL,
            conscientiousness REAL,
            extraversion REAL,
            agreeableness REAL,
            neuroticism REAL,
            method TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_analysis(image_name: str, scores: dict, method: str):
    conn = _get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO analyses (timestamp, image_name, openness, conscientiousness, extraversion, agreeableness, neuroticism, method)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        image_name,
        scores.get("Openness", 0),
        scores.get("Conscientiousness", 0),
        scores.get("Extraversion", 0),
        scores.get("Agreeableness", 0),
        scores.get("Neuroticism", 0),
        method
    ))
    conn.commit()
    conn.close()

def get_history():
    conn = _get_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM analyses ORDER BY timestamp DESC')
    rows = c.fetchall()
    columns = [desc[0] for desc in c.description]
    conn.close()
    return [dict(zip(columns, row)) for row in rows]

# Initialize db on import
init_db()
