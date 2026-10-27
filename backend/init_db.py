"""Create any missing SmartAttend tables without modifying existing rows."""

from backend.database import engine
from backend.migrations import upgrade_database


if __name__ == "__main__":
    upgrade_database(engine)
    print("SmartAttend tables and additive schema updates are ready. Existing rows were preserved.")
