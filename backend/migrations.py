"""Safe additive schema updates for existing SmartAttend SQLite databases."""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from backend.models import Base


FACE_EMBEDDING_ADDITIONS = {
    "model_name": "VARCHAR NOT NULL DEFAULT 'ArcFace'",
    "model_version": "VARCHAR NOT NULL DEFAULT 'legacy-unspecified'",
    "vector_dimension": "INTEGER NOT NULL DEFAULT 512",
    "preprocessing": "VARCHAR NOT NULL DEFAULT 'legacy-unspecified'",
}

ATTENDANCE_RECORD_ADDITIONS = {
    "recognition_method": "VARCHAR NOT NULL DEFAULT 'legacy'",
    "recognition_distance": "FLOAT",
    "marked_by_user_id": "INTEGER REFERENCES users(id)",
}


def _backup_sqlite_database(engine: Engine) -> Path | None:
    database_path = engine.url.database
    if not database_path or database_path == ":memory:":
        return None
    source_path = Path(database_path).resolve()
    if not source_path.is_file():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = source_path.with_name(f"{source_path.name}.pre-migration-{stamp}.bak")
    suffix = 1
    while backup_path.exists():
        backup_path = source_path.with_name(
            f"{source_path.name}.pre-migration-{stamp}-{suffix}.bak"
        )
        suffix += 1
    with sqlite3.connect(source_path) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)
        result = target.execute("PRAGMA integrity_check").fetchone()
        if not result or result[0] != "ok":
            target.close()
            backup_path.unlink(missing_ok=True)
            raise RuntimeError("SQLite pre-migration backup failed integrity_check")
    return backup_path


def upgrade_database(engine: Engine) -> None:
    """Apply additive biometric, account-link, and subject-assignment updates."""
    if engine.dialect.name != "sqlite":
        Base.metadata.create_all(bind=engine)
        return

    inspector = inspect(engine)
    tables_before = set(inspector.get_table_names())
    needs_update = bool(tables_before and "attendance_audits" not in tables_before)
    if "users" in tables_before:
        columns = {column["name"] for column in inspector.get_columns("users")}
        needs_update = needs_update or bool({"username", "student_id", "teacher_id", "must_change_password"} - columns)
    if "subjects" in tables_before:
        columns = {column["name"] for column in inspector.get_columns("subjects")}
        needs_update = needs_update or "teacher_id" not in columns
    if "face_embeddings" in tables_before:
        columns = {column["name"] for column in inspector.get_columns("face_embeddings")}
        needs_update = needs_update or bool(set(FACE_EMBEDDING_ADDITIONS) - columns)
    if "attendance_records" in tables_before:
        columns = {column["name"] for column in inspector.get_columns("attendance_records")}
        needs_update = needs_update or bool(set(ATTENDANCE_RECORD_ADDITIONS) - columns)
    if needs_update:
        _backup_sqlite_database(engine)

    Base.metadata.create_all(bind=engine)

    with engine.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        columns = {column["name"] for column in inspect(connection).get_columns("face_embeddings")}
        for name, definition in FACE_EMBEDDING_ADDITIONS.items():
            if name not in columns:
                connection.exec_driver_sql(
                    f"ALTER TABLE face_embeddings ADD COLUMN {name} {definition}"
                )
        columns = {column["name"] for column in inspect(connection).get_columns("attendance_records")}
        for name, definition in ATTENDANCE_RECORD_ADDITIONS.items():
            if name not in columns:
                connection.exec_driver_sql(
                    f"ALTER TABLE attendance_records ADD COLUMN {name} {definition}"
                )
        columns = {column["name"] for column in inspect(connection).get_columns("users")}
        for name, definition in {
            "username": "VARCHAR",
            "student_id": "INTEGER REFERENCES students(id)",
            "teacher_id": "INTEGER REFERENCES teachers(id)",
            "must_change_password": "BOOLEAN NOT NULL DEFAULT 0",
        }.items():
            if name not in columns:
                connection.exec_driver_sql(f"ALTER TABLE users ADD COLUMN {name} {definition}")
        connection.exec_driver_sql("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_username_unique ON users(lower(username))")
        connection.exec_driver_sql("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_student_id_unique ON users(student_id)")
        connection.exec_driver_sql("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_teacher_id_unique ON users(teacher_id)")
        subject_columns = {column["name"] for column in inspect(connection).get_columns("subjects")}
        if "teacher_id" not in subject_columns:
            connection.exec_driver_sql("ALTER TABLE subjects ADD COLUMN teacher_id INTEGER REFERENCES teachers(id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_subjects_teacher_id ON subjects(teacher_id)")
        connection.exec_driver_sql(
            "UPDATE users SET student_id=(SELECT students.id FROM students "
            "WHERE lower(students.email)=lower(users.email)) "
            "WHERE users.role='student' AND users.student_id IS NULL "
            "AND (SELECT COUNT(*) FROM students WHERE lower(students.email)=lower(users.email))=1 "
            "AND NOT EXISTS (SELECT 1 FROM users linked WHERE linked.student_id="
            "(SELECT students.id FROM students WHERE lower(students.email)=lower(users.email)))"
        )
        connection.exec_driver_sql(
            "UPDATE users SET teacher_id=(SELECT teachers.id FROM teachers "
            "WHERE lower(teachers.email)=lower(users.email)) "
            "WHERE users.role='teacher' AND users.teacher_id IS NULL "
            "AND (SELECT COUNT(*) FROM teachers WHERE lower(teachers.email)=lower(users.email))=1 "
            "AND NOT EXISTS (SELECT 1 FROM users linked WHERE linked.teacher_id="
            "(SELECT teachers.id FROM teachers WHERE lower(teachers.email)=lower(users.email)))"
        )
        connection.commit()
