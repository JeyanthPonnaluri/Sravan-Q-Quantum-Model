import sqlite3

# Connect to the database
db_path = r"d:\Sravan-Q-Quantum-Model\fraud_detection.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Query the transactions table
cursor.execute("SELECT * FROM transactions")
rows = cursor.fetchall()

# Print the table contents
columns = [description[0] for description in cursor.description]
print(f"{' | '.join(columns)}")
print("-" * 80)
for row in rows:
    print(" | ".join(map(str, row)))

conn.close()