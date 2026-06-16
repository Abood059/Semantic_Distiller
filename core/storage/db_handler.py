import sqlite3
from pathlib import Path
from typing import List
import numpy as np
from array import array


class DatabaseHandler:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with WAL mode enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
    
    def initialize_db(self) -> None:
        """Create the results table if it doesn't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                layer INTEGER NOT NULL,
                node INTEGER NOT NULL,
                sentence_index INTEGER NOT NULL,
                sentence TEXT NOT NULL,
                embedding BLOB,
                status TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (layer, node, sentence_index)
            )
        """)
        conn.commit()
        conn.close()
    
    def _serialize_embedding(self, embedding_list: List[float]) -> bytes:
        """Convert embedding list to bytes using array('f')."""
        arr = array('f', embedding_list)
        return arr.tobytes()
    
    def _deserialize_embedding(self, blob: bytes) -> List[float]:
        """Convert bytes back to embedding list using numpy."""
        return np.frombuffer(blob, dtype=np.float32).tolist()
    
    def save_node_results(self, layer: int, node: int, sentences: List[str], embeddings: List[List[float]]) -> None:
        """Save node results to database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        for idx, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            embedding_blob = self._serialize_embedding(embedding)
            cursor.execute("""
                INSERT OR REPLACE INTO results 
                (layer, node, sentence_index, sentence, embedding, status)
                VALUES (?, ?, ?, ?, ?, 'DONE')
            """, (layer, node, idx, sentence, embedding_blob))
        
        conn.commit()
        conn.close()
    
    def get_inputs_for_node(self, layer: int, node: int, num_nodes: int) -> List[str]:
        """Get input sentences for a node from the previous layer. # FIXED: Returns ONE sentence per previous node where sentence_index = node - 1."""
        if layer == 0:
            return []
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        sentences = []
        for prev_node in range(num_nodes):
            cursor.execute("""
                SELECT sentence FROM results
                WHERE layer = ? AND node = ? AND sentence_index = ? AND status = 'DONE'
            """, (layer - 1, prev_node, node - 1))  # FIXED: Use sentence_index = node - 1 to get one specific sentence
            row = cursor.fetchone()  # FIXED: Use fetchone() instead of fetchall()
            sentences.append(row[0] if row else "")  # FIXED: Append single sentence or empty string
        
        conn.close()
        return sentences
