"""Script to empty all tables in embedded_docs.db"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedded_docs.db")


def clear_tables():
    if not os.path.exists(DB_PATH):
        print(f"Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all user tables
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
    )
    tables = [row[0] for row in cursor.fetchall()]

    print(f"Database: {DB_PATH}")
    print(f"Tables found: {tables}\n")

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} rows -> deleting...", end=" ")
        cursor.execute(f"DELETE FROM [{table}]")
        print("done")

    # Reset autoincrement counters
    cursor.execute("DELETE FROM sqlite_sequence")

    conn.commit()

    # Reclaim disk space
    cursor.execute("VACUUM")

    conn.close()
    print("\nAll tables emptied successfully.")


if __name__ == "__main__":
    confirm = input("This will DELETE all data from embedded_docs.db. Continue? (yes/no): ")
    if confirm.strip().lower() == "yes":
        clear_tables()
    else:
        print("Aborted.")
