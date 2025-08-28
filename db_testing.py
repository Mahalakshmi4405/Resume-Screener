import sqlite3
import pandas as pd
from tabulate import tabulate
import os

def export_resumes():
    db_path = os.path.abspath("resumes.db")
    print(f"\n🔗 Connecting to database: {db_path}")

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all data
    cursor.execute("SELECT * FROM resumes")
    rows = cursor.fetchall()

    # Get column names
    col_names = [description[0] for description in cursor.description]

    if rows:
        # Print in table format
        print("\n📄 Resumes Table:\n")
        print(tabulate(rows, headers=col_names, tablefmt="grid"))

        # Save to CSV
        df = pd.DataFrame(rows, columns=col_names)
        df.to_csv("resumes.csv", index=False)
        print("\n✅ Data exported to resumes.csv")
    else:
        print("\n⚠️ No data found in the resumes table.")

    # Close connection
    conn.close()

if __name__ == "__main__":
    export_resumes()
