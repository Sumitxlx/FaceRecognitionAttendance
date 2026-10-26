from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# SQLite database file
DATABASE_PATH = BASE_DIR / "smartattend.db"

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"


# Create database engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)


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