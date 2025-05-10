import sqlite3

def init_db():
    conn = sqlite3.connect("creatorguard.db")
    cursor = conn.cursor()

    with open("schema.sql", "r") as f:
        schema = f.read()
        cursor.executescript(schema)

    conn.commit()
    conn.close()
    print("✅ Database initialized: creatorguard.db")

if __name__ == "__main__":
    init_db()
