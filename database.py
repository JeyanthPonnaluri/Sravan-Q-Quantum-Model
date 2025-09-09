"""
Database management for fraud detection system with blockchain demo
"""
import sqlite3
import hashlib
import json
import time
from datetime import datetime
from typing import List, Dict, Optional

class Database:
    def __init__(self, db_path: str = "fraud_detection.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_hash TEXT UNIQUE NOT NULL,
                amount REAL NOT NULL,
                hour_of_day INTEGER NOT NULL,
                is_weekend INTEGER NOT NULL,
                day_of_week TEXT NOT NULL,
                sender_age_group TEXT NOT NULL,
                receiver_age_group TEXT NOT NULL,
                sender_state TEXT NOT NULL,
                sender_bank TEXT NOT NULL,
                receiver_bank TEXT NOT NULL,
                merchant_category TEXT NOT NULL,
                device_type TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                network_type TEXT NOT NULL,
                transaction_status TEXT NOT NULL,
                quantum_score REAL,
                classical_score REAL,
                logical_score REAL,
                fusion_score REAL,
                risk_level TEXT,
                confidence TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                block_hash TEXT
            )
        """)
        
        # Blockchain blocks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blockchain_blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_hash TEXT UNIQUE NOT NULL,
                previous_hash TEXT NOT NULL,
                merkle_root TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                nonce INTEGER NOT NULL,
                difficulty INTEGER DEFAULT 4,
                transactions_count INTEGER NOT NULL,
                block_data TEXT NOT NULL
            )
        """)
        
        # Create genesis block if blockchain is empty
        cursor.execute("SELECT COUNT(*) FROM blockchain_blocks")
        if cursor.fetchone()[0] == 0:
            self.create_genesis_block(cursor)
        
        conn.commit()
        conn.close()
    
    def create_genesis_block(self, cursor):
        """Create the genesis block for the blockchain"""
        genesis_data = {
            "transactions": [],
            "message": "Genesis Block - Fraud Detection Blockchain"
        }
        
        genesis_hash = self.calculate_hash("0", "", int(time.time()), 0, json.dumps(genesis_data))
        
        cursor.execute("""
            INSERT INTO blockchain_blocks 
            (block_hash, previous_hash, merkle_root, nonce, transactions_count, block_data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (genesis_hash, "0", "0", 0, 0, json.dumps(genesis_data)))
    
    def calculate_hash(self, previous_hash: str, merkle_root: str, timestamp: int, nonce: int, data: str) -> str:
        """Calculate SHA-256 hash for a block"""
        block_string = f"{previous_hash}{merkle_root}{timestamp}{nonce}{data}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def calculate_merkle_root(self, transaction_hashes: List[str]) -> str:
        """Calculate Merkle root from transaction hashes"""
        if not transaction_hashes:
            return "0"
        
        if len(transaction_hashes) == 1:
            return transaction_hashes[0]
        
        # Simple implementation - in production, use proper Merkle tree
        combined = "".join(sorted(transaction_hashes))
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def mine_block(self, transactions: List[Dict], difficulty: int = 4) -> Dict:
        """Mine a new block with proof-of-work"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get previous block
        cursor.execute("SELECT block_hash FROM blockchain_blocks ORDER BY id DESC LIMIT 1")
        previous_hash = cursor.fetchone()[0]
        
        # Calculate merkle root
        transaction_hashes = [tx.get('transaction_hash', '') for tx in transactions]
        merkle_root = self.calculate_merkle_root(transaction_hashes)
        
        # Prepare block data
        block_data = {
            "transactions": transactions,
            "mined_at": datetime.now().isoformat()
        }
        
        # Mining (proof of work)
        nonce = 0
        timestamp = int(time.time())
        target = "0" * difficulty
        
        while True:
            block_hash = self.calculate_hash(previous_hash, merkle_root, timestamp, nonce, json.dumps(block_data))
            if block_hash.startswith(target):
                break
            nonce += 1
            
            # Prevent infinite loop in demo
            if nonce > 100000:
                break
        
        # Save block to database
        cursor.execute("""
            INSERT INTO blockchain_blocks 
            (block_hash, previous_hash, merkle_root, nonce, transactions_count, block_data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (block_hash, previous_hash, merkle_root, nonce, len(transactions), json.dumps(block_data)))
        
        # Update transactions with block hash
        for tx in transactions:
            cursor.execute("""
                UPDATE transactions SET block_hash = ? WHERE transaction_hash = ?
            """, (block_hash, tx.get('transaction_hash')))
        
        conn.commit()
        conn.close()
        
        return {
            "block_hash": block_hash,
            "previous_hash": previous_hash,
            "merkle_root": merkle_root,
            "nonce": nonce,
            "transactions_count": len(transactions),
            "difficulty": difficulty
        }
    
    def save_transaction(self, transaction_data: Dict) -> str:
        """Save transaction to database and return transaction hash"""
        # Generate transaction hash
        tx_string = json.dumps(transaction_data, sort_keys=True)
        transaction_hash = hashlib.sha256(tx_string.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO transactions (
                    transaction_hash, amount, hour_of_day, is_weekend, day_of_week,
                    sender_age_group, receiver_age_group, sender_state, sender_bank,
                    receiver_bank, merchant_category, device_type, transaction_type,
                    network_type, transaction_status, quantum_score, classical_score,
                    logical_score, fusion_score, risk_level, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transaction_hash,
                transaction_data['amount'],
                transaction_data['hour_of_day'],
                transaction_data['is_weekend'],
                transaction_data['day_of_week'],
                transaction_data['sender_age_group'],
                transaction_data['receiver_age_group'],
                transaction_data['sender_state'],
                transaction_data['sender_bank'],
                transaction_data['receiver_bank'],
                transaction_data['merchant_category'],
                transaction_data['device_type'],
                transaction_data['transaction_type'],
                transaction_data['network_type'],
                transaction_data['transaction_status'],
                transaction_data.get('quantum_score'),
                transaction_data.get('classical_score'),
                transaction_data.get('logical_score'),
                transaction_data.get('fusion_score'),
                transaction_data.get('risk_level'),
                transaction_data.get('confidence')
            ))
            
            conn.commit()
            return transaction_hash
            
        except sqlite3.IntegrityError:
            # Transaction already exists
            return transaction_hash
        finally:
            conn.close()
    
    def get_recent_transactions(self, limit: int = 10) -> List[Dict]:
        """Get recent transactions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM transactions 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        columns = [description[0] for description in cursor.description]
        transactions = []
        
        for row in cursor.fetchall():
            transactions.append(dict(zip(columns, row)))
        
        conn.close()
        return transactions
    
    def get_blockchain_info(self) -> Dict:
        """Get blockchain statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM blockchain_blocks")
        total_blocks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM transactions")
        total_transactions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM transactions WHERE block_hash IS NOT NULL")
        confirmed_transactions = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT block_hash, timestamp, transactions_count 
            FROM blockchain_blocks 
            ORDER BY id DESC 
            LIMIT 5
        """)
        
        recent_blocks = []
        for row in cursor.fetchall():
            recent_blocks.append({
                "hash": row[0][:16] + "...",
                "timestamp": row[1],
                "transactions": row[2]
            })
        
        conn.close()
        
        return {
            "total_blocks": total_blocks,
            "total_transactions": total_transactions,
            "confirmed_transactions": confirmed_transactions,
            "pending_transactions": total_transactions - confirmed_transactions,
            "recent_blocks": recent_blocks
        }
    
    def get_pending_transactions(self) -> List[Dict]:
        """Get transactions not yet in a block"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM transactions 
            WHERE block_hash IS NULL 
            ORDER BY timestamp ASC
        """)
        
        columns = [description[0] for description in cursor.description]
        transactions = []
        
        for row in cursor.fetchall():
            transactions.append(dict(zip(columns, row)))
        
        conn.close()
        return transactions