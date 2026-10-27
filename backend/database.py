from pathlib import Path
import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# SQLite database file
DATABASE_PATH = BASE_DIR / "smartattend.db"

DATABASE_URL = os.environ.get("SMARTATTEND_DATABASE_URL", f"sqlite:///{DATABASE_PATH}")


# Create database engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30}
)


@event.listens_for(engine, "connect")
def enable_sqlite_foreign_keys(connection, _record):
    if engine.dialect.name == "sqlite":
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# Create database sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# Dependency for getting a database session
def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
