import tempfile
from pathlib import Path
import sqlite3
import unittest

from sqlalchemy import create_engine, inspect

from backend.migrations import upgrade_database


class SchemaMigrationTests(unittest.TestCase):
    def test_legacy_student_account_is_linked_without_replacing_data(self):
        with tempfile.TemporaryDirectory(prefix="smartattend-account-migration-") as folder:
            path = Path(folder) / "legacy.sqlite"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR NOT NULL UNIQUE, password_hash VARCHAR NOT NULL, role VARCHAR NOT NULL)")
                connection.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL UNIQUE)")
                connection.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, roll_number VARCHAR NOT NULL UNIQUE, email VARCHAR NOT NULL UNIQUE, department_id INTEGER NOT NULL)")
                connection.execute("CREATE TABLE teachers (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, email VARCHAR NOT NULL UNIQUE, department_id INTEGER NOT NULL)")
                connection.execute("CREATE TABLE subjects (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, code VARCHAR NOT NULL UNIQUE, department_id INTEGER NOT NULL)")
                connection.execute("INSERT INTO departments VALUES (1,'Engineering')")
                connection.execute("INSERT INTO students VALUES (7,'Student','R7','student@example.com',1)")
                connection.execute("INSERT INTO teachers VALUES (8,'Teacher','teacher@example.com',1)")
                connection.execute("INSERT INTO subjects VALUES (9,'Existing subject','ENG1',1)")
                connection.execute("INSERT INTO users VALUES (3,'student@example.com','existing-hash','student')")
                connection.execute("INSERT INTO users VALUES (4,'teacher@example.com','teacher-hash','teacher')")
            engine = create_engine(f"sqlite:///{path}")
            try:
                upgrade_database(engine)
                with engine.connect() as connection:
                    user = connection.exec_driver_sql("SELECT email,password_hash,role,student_id,must_change_password FROM users WHERE id=3").one()
                    teacher_user = connection.exec_driver_sql("SELECT teacher_id FROM users WHERE id=4").scalar_one()
                    subject = connection.exec_driver_sql("SELECT teacher_id FROM subjects WHERE id=9").scalar_one()
                self.assertEqual(tuple(user[:3]), ("student@example.com", "existing-hash", "student"))
                self.assertEqual(user.student_id, 7)
                self.assertFalse(user.must_change_password)
                self.assertEqual(teacher_user, 8)
                self.assertIsNone(subject)
                backups = list(Path(folder).glob("legacy.sqlite.pre-migration-*.bak"))
                self.assertEqual(len(backups), 1)
            finally:
                engine.dispose()

    def test_legacy_embedding_rows_are_preserved_and_get_metadata(self):
        with tempfile.TemporaryDirectory(prefix="smartattend-migration-test-") as folder:
            path = Path(folder) / "legacy.sqlite"
            with sqlite3.connect(path) as connection:
                connection.execute(
                    "CREATE TABLE face_embeddings ("
                    "id INTEGER PRIMARY KEY, student_id INTEGER NOT NULL UNIQUE, "
                    "embedding BLOB NOT NULL, created_at DATETIME NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE attendance_records ("
                    "id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, student_id INTEGER NOT NULL, "
                    "status VARCHAR NOT NULL, confidence FLOAT, marked_at DATETIME NOT NULL, "
                    "CONSTRAINT unique_student_per_session UNIQUE(session_id, student_id))"
                )
                original = bytes(range(64))
                connection.execute(
                    "INSERT INTO face_embeddings (id,student_id,embedding,created_at) VALUES (1,7,?,?)",
                    (original, "2026-01-01 00:00:00"),
                )
                connection.execute(
                    "INSERT INTO attendance_records (id,session_id,student_id,status,marked_at) "
                    "VALUES (11,3,7,'present','2026-01-01 10:00:00')"
                )

            engine = create_engine(f"sqlite:///{path}")
            try:
                upgrade_database(engine)
                names = {column["name"] for column in inspect(engine).get_columns("face_embeddings")}
                self.assertTrue({"model_name", "model_version", "vector_dimension", "preprocessing"} <= names)
                attendance_columns = {column["name"] for column in inspect(engine).get_columns("attendance_records")}
                self.assertTrue({"recognition_method", "recognition_distance", "marked_by_user_id"} <= attendance_columns)
                with engine.connect() as connection:
                    row = connection.exec_driver_sql(
                        "SELECT embedding, model_name, model_version, vector_dimension, preprocessing "
                        "FROM face_embeddings WHERE id=1"
                    ).one()
                self.assertEqual(row.embedding, original)
                self.assertEqual(row.model_name, "ArcFace")
                self.assertEqual(row.model_version, "legacy-unspecified")
                self.assertEqual(row.vector_dimension, 512)
                self.assertEqual(row.preprocessing, "legacy-unspecified")
                with engine.connect() as connection:
                    attendance = connection.exec_driver_sql(
                        "SELECT status, recognition_method, recognition_distance, marked_by_user_id "
                        "FROM attendance_records WHERE id=11"
                    ).one()
                self.assertEqual(attendance.status, "present")
                self.assertEqual(attendance.recognition_method, "legacy")
                self.assertIsNone(attendance.recognition_distance)
                self.assertIsNone(attendance.marked_by_user_id)
                backups = list(Path(folder).glob("legacy.sqlite.pre-migration-*.bak"))
                self.assertEqual(len(backups), 1)
                with sqlite3.connect(backups[0]) as backup:
                    self.assertEqual(backup.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                    original_row = backup.execute(
                        "SELECT embedding FROM face_embeddings WHERE id=1"
                    ).fetchone()
                    attendance_row = backup.execute(
                        "SELECT status FROM attendance_records WHERE id=11"
                    ).fetchone()
                self.assertEqual(original_row[0], original)
                self.assertEqual(attendance_row[0], "present")
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
