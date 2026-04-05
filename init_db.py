import sqlite3
import os

DB_PATH = 'database.db'

def init_db():
    if os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} already exists.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create datasets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create decisions table for tracking individual entries
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gender TEXT,
            age INTEGER,
            experience TEXT,
            decision TEXT
        )
    ''')

    # Create audit_results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bias_score REAL,
            ethical_risk REAL,
            explanation TEXT,
            recommendation TEXT,
            audit_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized successfully.")

if __name__ == '__main__':
    init_db()
