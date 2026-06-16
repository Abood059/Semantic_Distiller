import sqlite3
from .db_handler import DatabaseHandler


def is_node_done(db_handler: DatabaseHandler, layer: int, node: int) -> bool:
    """Check if a node is already completed."""
    conn = db_handler._get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM results
        WHERE layer = ? AND node = ? AND status = 'DONE'
    """, (layer, node))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def mark_node_processing(db_handler: DatabaseHandler, layer: int, node: int) -> None:
    """Mark a node as processing."""
    conn = db_handler._get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE results
        SET status = 'PROCESSING'
        WHERE layer = ? AND node = ?
    """, (layer, node))
    conn.commit()
    conn.close()


def reset_failed_nodes(db_handler: DatabaseHandler, layer: int) -> None:
    """Reset failed nodes from PROCESSING to PENDING."""
    conn = db_handler._get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE results
        SET status = 'PENDING'
        WHERE layer = ? AND status = 'PROCESSING'
    """, (layer,))
    conn.commit()
    conn.close()
