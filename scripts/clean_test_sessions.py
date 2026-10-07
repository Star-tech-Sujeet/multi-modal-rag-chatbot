#!/usr/bin/env python3
"""
Safe Cleanup Utility for Aura AI Persistent Sessions Database.

Identifies and removes test artifacts ('Corrupt Test', 'Text Only', empty test chats)
while strictly preserving all genuine user conversation history.

Features:
- Safe dry-run mode by default (use --confirm to execute)
- Automatic full database backup before deletion
- Foreign-key cascading cleanup of messages
- SQLite VACUUM to compact database file
"""

import argparse
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple

KNOWN_TEST_TITLES = {
    "Corrupt Test",
    "Text Only",
    "API Test Session",
    "Turn Test",
    "Financial Review 2024",
    "Updated API Title",
}


def is_test_session(title: str, msg_count: int, metadata_str: str) -> Tuple[bool, str]:
    """
    Determine if a session is a test artifact vs a genuine user conversation.
    Returns (is_test, reason).
    """
    title_clean = (title or "").strip()

    # 1. Exact match with known automated test titles
    if title_clean in KNOWN_TEST_TITLES:
        return True, f"Matches automated test title: '{title_clean}'"

    # 2. Check metadata markers
    if metadata_str:
        try:
            meta = json.loads(metadata_str)
            if isinstance(meta, dict):
                if meta.get("is_test") or meta.get("test_mode") or meta.get("tag") == "finance":
                    return True, f"Metadata indicates test artifact: {meta}"
        except Exception:
            pass

    # 3. Empty 'New Chat' sessions with 0 messages
    if title_clean == "New Chat" and msg_count == 0:
        return True, "Empty 'New Chat' session with 0 messages"

    # Genuine conversation
    return False, "Genuine user conversation"


def analyze_sessions(db_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Analyze all sessions in the database and classify into test artifacts and genuine sessions.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT s.session_id, s.title, s.created_at, s.updated_at, s.metadata,
                   COUNT(m.message_id) as msg_count
            FROM sessions s
            LEFT JOIN messages m ON s.session_id = m.session_id
            GROUP BY s.session_id
            ORDER BY s.created_at DESC
        """).fetchall()

        to_delete = []
        to_preserve = []

        for r in rows:
            item = {
                "session_id": r["session_id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "msg_count": r["msg_count"],
                "metadata": r["metadata"],
            }
            test_flag, reason = is_test_session(r["title"], r["msg_count"], r["metadata"])
            item["reason"] = reason

            if test_flag:
                to_delete.append(item)
            else:
                to_preserve.append(item)

        return to_delete, to_preserve
    finally:
        conn.close()


def clean_test_sessions(db_path: str = "./data/sessions.sqlite", confirm: bool = False) -> Dict[str, Any]:
    """
    Execute session analysis and optional cleanup.
    """
    to_delete, to_preserve = analyze_sessions(db_path)

    print("=" * 70)
    print("Aura AI — Session Database Audit & Cleanup")
    print("=" * 70)
    print(f"Database: {os.path.abspath(db_path)}")
    print(f"Total sessions found: {len(to_delete) + len(to_preserve)}")
    print(f"  - Test artifacts to clean: {len(to_delete)}")
    print(f"  - Genuine conversations to preserve: {len(to_preserve)}")
    print("-" * 70)

    if to_preserve:
        print("\n[PRESERVED SESSIONS]")
        for s in to_preserve:
            print(f"  ID: {s['session_id']} | Title: '{s['title']}' | Messages: {s['msg_count']} | Created: {s['created_at']}")

    if to_delete:
        print(f"\n[TEST ARTIFACTS ({len(to_delete)} total)]")
        title_counts = {}
        for s in to_delete:
            title_counts[s["title"]] = title_counts.get(s["title"], 0) + 1
        for t, c in title_counts.items():
            print(f"  - '{t}': {c} sessions")

    if not confirm:
        print("\n" + "=" * 70)
        print("DRY RUN COMPLETE — No data was modified.")
        print("To safely delete test artifacts and create an automatic backup, run:")
        print("  python scripts/clean_test_sessions.py --confirm")
        print("=" * 70)
        return {
            "status": "dry_run",
            "test_sessions_count": len(to_delete),
            "preserved_sessions_count": len(to_preserve),
        }

    if not to_delete:
        print("\nNo test artifacts to delete. Database is already clean.")
        return {
            "status": "already_clean",
            "test_sessions_count": 0,
            "preserved_sessions_count": len(to_preserve),
        }

    # Backup database before deletion
    backup_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{backup_ts}"
    shutil.copy2(db_path, backup_path)
    shutil.copy2(db_path, f"{db_path}.bak")
    print(f"\nDatabase backed up to:")
    print(f"  {backup_path}")
    print(f"  {db_path}.bak")

    # Perform atomic deletion
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        cursor = conn.cursor()
        ids_to_delete = [s["session_id"] for s in to_delete]
        chunk_size = 500
        deleted_count = 0
        for i in range(0, len(ids_to_delete), chunk_size):
            chunk = ids_to_delete[i:i+chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            cursor.execute(f"DELETE FROM sessions WHERE session_id IN ({placeholders})", chunk)
            deleted_count += cursor.rowcount

        conn.commit()
        conn.execute("VACUUM;")
        print(f"\nSuccessfully deleted {deleted_count} test sessions (and cascaded messages).")
    except Exception as e:
        conn.rollback()
        print(f"\nError during deletion: {e}")
        raise
    finally:
        conn.close()

    # Verification
    remaining_delete, remaining_preserve = analyze_sessions(db_path)
    print(f"\nVerification:")
    print(f"  Remaining sessions: {len(remaining_preserve)}")
    for s in remaining_preserve:
        print(f"  - '{s['title']}' (ID: {s['session_id']}, Messages: {s['msg_count']})")
    print("=" * 70)

    return {
        "status": "cleaned",
        "deleted_count": len(to_delete),
        "preserved_count": len(remaining_preserve),
        "backup_path": backup_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean test session artifacts from Aura AI SQLite database.")
    parser.add_argument("--db-path", default="./data/sessions.sqlite", help="Path to SQLite database")
    parser.add_argument("--confirm", action="store_true", help="Confirm deletion of test artifacts")
    args = parser.parse_args()

    clean_test_sessions(args.db_path, confirm=args.confirm)
