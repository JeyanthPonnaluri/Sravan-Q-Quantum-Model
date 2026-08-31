"""
Database management for fraud detection system with blockchain demo (Supports Supabase PostgreSQL & SQLite Fallback)
"""
import os
import sqlite3
import hashlib
import json
import time
from datetime import datetime
from typing import List, Dict, Optional

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

class Database:
    def __init__(self, db_path: str = "fraud_detection.db"):
        self.db_path = db_path
        self.database_url = os.getenv("DATABASE_URL")
        self.use_postgres = False
        
        # Load local .env file manually if exists
        if not self.database_url and os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    if "=" in line and not line.startswith("#"):
                        parts = line.strip().split("=", 1)
                        if len(parts) == 2 and parts[0].strip() == "DATABASE_URL":
                            self.database_url = parts[1].strip()
                            break

        if PSYCOPG2_AVAILABLE and self.database_url:
            try:
                # Test connection to PostgreSQL
                conn = psycopg2.connect(self.database_url, connect_timeout=5)
                conn.close()
                self.use_postgres = True
                print("[Database] Successfully connected to Supabase PostgreSQL backend.")
            except Exception as e:
                print(f"[Database] Warning: Failed to connect to PostgreSQL ({e}). Falling back to SQLite.")
        else:
            if not PSYCOPG2_AVAILABLE:
                print("[Database] psycopg2 module not available. Falling back to SQLite.")
            else:
                print("[Database] DATABASE_URL not set. Falling back to SQLite.")
                
        self.init_database()
    
    def _get_connection(self):
        if self.use_postgres:
            conn = psycopg2.connect(self.database_url)
            conn.autocommit = True
            return conn
        else:
            return sqlite3.connect(self.db_path)

    def _execute(self, cursor, query: str, params=None):
        if params is None:
            params = ()
        if self.use_postgres:
            # Translate SQLite '?' to PostgreSQL '%s'
            query_translated = query.replace('?', '%s')
            cursor.execute(query_translated, params)
        else:
            cursor.execute(query, params)

    def init_database(self):
        """Initialize database with required tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if self.use_postgres:
            # PostgreSQL Tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id SERIAL PRIMARY KEY,
                    transaction_hash VARCHAR(255) UNIQUE NOT NULL,
                    amount DOUBLE PRECISION NOT NULL,
                    hour_of_day INTEGER NOT NULL,
                    is_weekend INTEGER NOT NULL,
                    day_of_week VARCHAR(50) NOT NULL,
                    sender_age_group VARCHAR(50) NOT NULL,
                    receiver_age_group VARCHAR(50) NOT NULL,
                    sender_state VARCHAR(100) NOT NULL,
                    sender_bank VARCHAR(100) NOT NULL,
                    receiver_bank VARCHAR(100) NOT NULL,
                    merchant_category VARCHAR(100) NOT NULL,
                    device_type VARCHAR(100) NOT NULL,
                    transaction_type VARCHAR(100) NOT NULL,
                    network_type VARCHAR(100) NOT NULL,
                    transaction_status VARCHAR(100) NOT NULL,
                    quantum_score DOUBLE PRECISION,
                    classical_score DOUBLE PRECISION,
                    logical_score DOUBLE PRECISION,
                    fusion_score DOUBLE PRECISION,
                    risk_level VARCHAR(50),
                    confidence VARCHAR(50),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    block_hash VARCHAR(255)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS blockchain_blocks (
                    id SERIAL PRIMARY KEY,
                    block_hash VARCHAR(255) UNIQUE NOT NULL,
                    previous_hash VARCHAR(255) NOT NULL,
                    merkle_root VARCHAR(255) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    nonce INTEGER NOT NULL,
                    difficulty INTEGER DEFAULT 4,
                    transactions_count INTEGER NOT NULL,
                    block_data TEXT NOT NULL
                )
            """)
            cursor.execute("SELECT COUNT(*) FROM blockchain_blocks")
            if cursor.fetchone()[0] == 0:
                self.create_genesis_block(cursor)
        else:
            # SQLite Tables
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
        
        self._execute(cursor, """
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
        
        combined = "".join(sorted(transaction_hashes))
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def mine_block(self, transactions: List[Dict], difficulty: int = 4) -> Dict:
        """Mine a new block with proof-of-work"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get previous block
        cursor.execute("SELECT block_hash FROM blockchain_blocks ORDER BY id DESC LIMIT 1")
        previous_hash = cursor.fetchone()[0]
        
        # Calculate merkle root
        transaction_hashes = [tx.get('transaction_hash', '') for tx in transactions]
        merkle_root = self.calculate_merkle_root(transaction_hashes)
        
        # Simple RSA modular exponentiation signatures for 3 virtual nodes
        nodes = {
            "node_primary": {"d": 79, "n": 3233},
            "node_validator_1": {"d": 101, "n": 2773},
            "node_validator_2": {"d": 125, "n": 2419}
        }
        peer_env = os.getenv("PEER_NODES", "")
        peer_nodes = [p.strip() for p in peer_env.split(",") if p.strip()]
        
        consensus_signatures = {}
        
        import urllib.request
        import urllib.error
        
        for node_id, keys in nodes.items():
            signature_acquired = False
            if peer_nodes:
                peer_idx = 0 if node_id == "node_validator_1" else (1 if node_id == "node_validator_2" else -1)
                if 0 <= peer_idx < len(peer_nodes):
                    peer_url = peer_nodes[peer_idx]
                    try:
                        req_url = f"{peer_url}/api/v1/consensus/sign"
                        req_data = json.dumps({"block_hash": previous_hash, "node_id": node_id}).encode('utf-8')
                        req = urllib.request.Request(
                            req_url, 
                            data=req_data, 
                            headers={'Content-Type': 'application/json'},
                            method='POST'
                        )
                        with urllib.request.urlopen(req, timeout=0.5) as response:
                            res = json.loads(response.read().decode('utf-8'))
                            if res.get("status") == "signed":
                                consensus_signatures[node_id] = res.get("signature")
                                print(f"[Consensus Network] Retrieved real validation signature from {node_id} ({peer_url})")
                                signature_acquired = True
                    except Exception:
                        pass
            
            if not signature_acquired:
                val = sum(ord(c) for c in previous_hash) % keys["n"]
                sig = pow(val, keys["d"], keys["n"])
                consensus_signatures[node_id] = f"{node_id}_sig_{sig}"

        block_data = {
            "transactions": transactions,
            "mined_at": datetime.now().isoformat(),
            "consensus_signatures": consensus_signatures
        }
        
        nonce = 0
        timestamp = int(time.time())
        target = "0" * difficulty
        
        while True:
            block_hash = self.calculate_hash(previous_hash, merkle_root, timestamp, nonce, json.dumps(block_data))
            if block_hash.startswith(target):
                break
            nonce += 1
            if nonce > 100000:
                break
        
        # Save block to database
        self._execute(cursor, """
            INSERT INTO blockchain_blocks 
            (block_hash, previous_hash, merkle_root, nonce, transactions_count, block_data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (block_hash, previous_hash, merkle_root, nonce, len(transactions), json.dumps(block_data)))
        
        # Update transactions with block hash
        for tx in transactions:
            self._execute(cursor, """
                UPDATE transactions SET block_hash = ? WHERE transaction_hash = ?
            """, (block_hash, tx.get('transaction_hash')))
        
        if not self.use_postgres:
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
        tx_string = json.dumps(transaction_data, sort_keys=True)
        transaction_hash = hashlib.sha256(tx_string.encode()).hexdigest()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            self._execute(cursor, """
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
            
            if not self.use_postgres:
                conn.commit()
            return transaction_hash
            
        except Exception:
            return transaction_hash
        finally:
            conn.close()
    
    def get_recent_transactions(self, limit: int = 10) -> List[Dict]:
        """Get recent transactions"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        self._execute(cursor, """
            SELECT * FROM transactions 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        columns = [description[0] for description in cursor.description]
        transactions = []
        
        for row in cursor.fetchall():
            # Convert timestamp to string if it is a datetime object
            row_list = list(row)
            for idx, col in enumerate(columns):
                if col == 'timestamp' and isinstance(row_list[idx], datetime):
                    row_list[idx] = row_list[idx].isoformat()
            transactions.append(dict(zip(columns, row_list)))
        
        conn.close()
        return transactions
    
    def get_blockchain_info(self) -> Dict:
        """Get blockchain statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM blockchain_blocks")
        total_blocks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM transactions")
        total_transactions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM transactions WHERE block_hash IS NOT NULL")
        confirmed_transactions = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT id, block_hash, previous_hash, merkle_root, timestamp, nonce, difficulty, transactions_count, block_data
            FROM blockchain_blocks 
            ORDER BY id ASC
        """)
        
        chain = []
        for row in cursor.fetchall():
            block_data_parsed = {}
            try:
                block_data_parsed = json.loads(row[8])
            except Exception:
                pass
            chain.append({
                "index": row[0],
                "hash": row[1],
                "previous_hash": row[2],
                "merkle_root": row[3],
                "timestamp": str(row[4]),
                "nonce": row[5],
                "difficulty": row[6],
                "transactions_count": row[7],
                "consensus_signatures": block_data_parsed.get("consensus_signatures", {})
            })
        
        conn.close()
        
        return {
            "total_blocks": total_blocks,
            "total_transactions": total_transactions,
            "confirmed_transactions": confirmed_transactions,
            "pending_transactions": total_transactions - confirmed_transactions,
            "chain": chain
        }
    
    def get_pending_transactions(self) -> List[Dict]:
        """Get transactions not yet in a block"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM transactions 
            WHERE block_hash IS NULL 
            ORDER BY timestamp ASC
        """)
        
        columns = [description[0] for description in cursor.description]
        transactions = []
        
        for row in cursor.fetchall():
            row_list = list(row)
            for idx, col in enumerate(columns):
                if col == 'timestamp' and isinstance(row_list[idx], datetime):
                    row_list[idx] = row_list[idx].isoformat()
            transactions.append(dict(zip(columns, row_list)))
        
        conn.close()
        return transactions