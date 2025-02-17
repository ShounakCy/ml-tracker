import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import yaml


def create_database():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    """Create the experiment tracking database if it doesn't exist"""
    db_name = config["database"]["name"]
    db_user = config["database"]["user"]
    db_password = config["database"]["password"]
    db_host = config["database"]["host"]

    # Connect to PostgreSQL server
    conn = psycopg2.connect(
        user=db_user, password=db_password, host=db_host, dbname="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    cur = conn.cursor()

    # Check if database exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    exists = cur.fetchone()

    if not exists:
        cur.execute(f"CREATE DATABASE {db_name}")
        print(f"Created database: {db_name}")

    cur.close()
    conn.close()

    # Return connection string for the new database
    return f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}"


def create_tables(conn_string):
    """Create the necessary tables for experiment tracking"""
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()

    # Create experiments table
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        config JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status VARCHAR(50) DEFAULT 'running',
        description TEXT,
        tags JSONB
    )
    """
    )

    # Create training_metrics table
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS training_metrics (
        metric_id SERIAL PRIMARY KEY,
        experiment_id INTEGER REFERENCES experiments(experiment_id),
        epoch INTEGER NOT NULL,
        metrics JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (experiment_id, epoch)
    )
    """
    )

    # Create evaluation_metrics table
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS evaluation_metrics (
        eval_id SERIAL PRIMARY KEY,
        experiment_id INTEGER REFERENCES experiments(experiment_id),
        metrics JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    )

    # Create artifacts table
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id SERIAL PRIMARY KEY,
        experiment_id INTEGER REFERENCES experiments(experiment_id),
        name VARCHAR(255) NOT NULL,
        type VARCHAR(50) NOT NULL,
        file_path VARCHAR(512) NOT NULL,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (experiment_id, name)
    )
    """
    )

    # Create indices for better query performance
    cur.execute(
        """
    CREATE INDEX IF NOT EXISTS idx_experiment_name ON experiments(name);
    CREATE INDEX IF NOT EXISTS idx_training_metrics_experiment ON training_metrics(experiment_id);
    CREATE INDEX IF NOT EXISTS idx_evaluation_metrics_experiment ON evaluation_metrics(experiment_id);
    CREATE INDEX IF NOT EXISTS idx_artifacts_experiment ON artifacts(experiment_id);
    """
    )

    conn.commit()
    cur.close()
    conn.close()


def upgrade(conn_string):
    """Add duration column to experiments table"""
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()

    cur.execute(
        """
    ALTER TABLE experiments
    ADD COLUMN IF NOT EXISTS duration_seconds INTEGER,
    ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP
    """
    )

    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    print("Setting up experiment tracking database...")
    conn_string = create_database()
    print("Creating tables...")
    create_tables(conn_string)
    print("Running migrations...")
    upgrade(conn_string)
    print("Database setup complete!")
