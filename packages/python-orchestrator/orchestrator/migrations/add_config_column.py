"""
Migration: Add config column to jobs table
Run this once to update existing database schema
"""
import psycopg2
import os
import time

def run_migration():
    """Add config JSON column to jobs table"""
    
    # Database connection parameters
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'postgres'),
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
        'database': os.getenv('POSTGRES_DB', 'reposynth'),
        'user': os.getenv('POSTGRES_USER', 'repouser'),
        'password': os.getenv('POSTGRES_PASSWORD', 'repopass123')
    }
    
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting to connect to database (attempt {attempt + 1}/{max_retries})...")
            conn = psycopg2.connect(**db_config)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Check if column already exists
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='jobs' AND column_name='config'
            """)
            
            if cursor.fetchone():
                print("✓ Column 'config' already exists in jobs table")
            else:
                # Add the config column
                print("Adding 'config' column to jobs table...")
                cursor.execute("""
                    ALTER TABLE jobs 
                    ADD COLUMN config JSON
                """)
                print("✓ Successfully added 'config' column to jobs table")
            
            cursor.close()
            conn.close()
            return True
            
        except psycopg2.OperationalError as e:
            if attempt < max_retries - 1:
                print(f"✗ Database not ready: {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"✗ Failed to connect to database after {max_retries} attempts")
                raise
        except Exception as e:
            print(f"✗ Migration failed: {e}")
            raise

if __name__ == '__main__':
    run_migration()
