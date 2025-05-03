#!/usr/bin/env python3
"""
Standalone script to reset the database without importing the app.
This directly manipulates the SQLite database to avoid schema conflicts.
"""
import os
import sys
import sqlite3
import dotenv

def reset_database():
    """Delete the database file and recreate it with the updated schema."""
    # Load environment variables from .env file
    dotenv.load_dotenv()
    
    # Get the database file path from environment variable or use default
    db_file = os.environ.get('DB_FILE', 'chatbot.db')
    
    # Check if the database file exists
    if os.path.exists(db_file):
        print(f"Removing existing database file: {db_file}")
        os.remove(db_file)
        print(f"Database file removed: {db_file}")
    else:
        print(f"Database file does not exist: {db_file}")
    
    # Also remove any -journal or -wal files
    for ext in ['-journal', '-wal', '-shm']:
        journal_file = f"{db_file}{ext}"
        if os.path.exists(journal_file):
            print(f"Removing journal file: {journal_file}")
            os.remove(journal_file)
    
    # Create a new SQLite database with the updated schema
    print("Creating new database with updated schema...")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create the tables with the updated schema
    cursor.executescript("""
    -- Create Wave table
    CREATE TABLE wave (
        id VARCHAR(255) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        author VARCHAR(255),
        serial_no INTEGER
    );
    
    -- Create Drop table with bot_replied_to column
    CREATE TABLE "drop" (
        id VARCHAR(255) PRIMARY KEY,
        wave_id VARCHAR(255) NOT NULL,
        author VARCHAR(255) NOT NULL,
        content TEXT,
        serial_no INTEGER,
        created_at TIMESTAMP,
        reply_to_id VARCHAR(255),
        bot_replied_to BOOLEAN DEFAULT 0,
        FOREIGN KEY (wave_id) REFERENCES wave (id),
        FOREIGN KEY (reply_to_id) REFERENCES "drop" (id)
    );
    
    -- Create WaveTracking table
    CREATE TABLE wave_tracking (
        wave_id VARCHAR(255) PRIMARY KEY,
        last_processed_serial_no INTEGER DEFAULT 0,
        last_interaction_serial_no INTEGER DEFAULT 0,
        accumulated_new_drops INTEGER DEFAULT 0,
        last_activity_check TIMESTAMP,
        FOREIGN KEY (wave_id) REFERENCES wave (id)
    );
    
    -- Create Identity table (replaces Author table)
    CREATE TABLE identities (
        id VARCHAR(255) PRIMARY KEY,
        handle VARCHAR(255) NOT NULL UNIQUE,
        normalized_handle VARCHAR(255),
        pfp VARCHAR(255),
        cic INTEGER,
        rep INTEGER,
        level INTEGER,
        tdh INTEGER,
        display VARCHAR(255),
        primary_wallet VARCHAR(255),
        pfp_url VARCHAR(255),
        profile_url VARCHAR(255)
    );
    """)

    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("Database reset complete!")

if __name__ == "__main__":
    # Confirm with the user
    print("WARNING: This will delete your existing database and recreate it with the updated schema.")
    print("All data will be lost. This should only be used during development.")
    
    confirm = input("Are you sure you want to continue? (y/N): ")
    if confirm.lower() != 'y':
        print("Database reset cancelled.")
        sys.exit(0)
    
    reset_database()
