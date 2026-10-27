import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
import tempfile
from pathlib import Path
from threading import Barrier

import numpy as np
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.auth import hash_password, verify_password
from backend.models import (
    AttendanceRecord,
    AttendanceSession,
    Base,
    Department,
    Student,
    Subject,
    Teacher,
    User,
)
from backend.token import ALGORITHM, SECRET_KEY, create_access_token
from jose import jwt


class CoreBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        @event.listens_for(self.engine, "connect")
        def enable_foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_password_hash_and_token(self):
        hashed = hash_password("a long test password")
        self.assertNotEqual(hashed, "a long test password")
        self.assertTrue(verify_password("a long test password", hashed))
        self.assertFalse(verify_password("wrong password", hashed))
        token = create_access_token({"sub": "17", "role": "teacher"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        self.assertEqual(payload["sub"], "17")
        self.assertEqual(payload["role"], "teacher")

    def test_embedding_float32_round_trip(self):
        expected = np.arange(512, dtype=np.float32)
        stored = expected.tobytes()
        restored = np.frombuffer(stored, dtype=np.float32)
        self.assertEqual(restored.shape, (512,))
        np.testing.assert_array_equal(restored, expected)

    def test_database_rejects_duplicate_attendance_for_session(self):
        department = Department(name="Test department")
        self.db.add(department)
        self.db.flush()
        teacher = Teacher(name="Test teacher", email="teacher@example.com", department_id=department.id)
        student = Student(name="Test student", roll_number="T1", email="student@example.com", department_id=department.id)
        subject = Subject(name="Test subject", code="TST1", department_id=department.id)
        self.db.add_all([teacher, student, subject])
        self.db.flush()
        session = AttendanceSession(
            teacher_id=teacher.id,
            subject_id=subject.id,
            session_date=date.today(),
            start_time=datetime.now(),
            status="active",
        )
        self.db.add(session)
        self.db.flush()
        self.db.add(AttendanceRecord(session_id=session.id, student_id=student.id, status="present"))
        self.db.commit()
        self.db.add(AttendanceRecord(session_id=session.id, student_id=student.id, status="present"))
        with self.assertRaises(IntegrityError):
            self.db.commit()

    def test_foreign_keys_reject_orphan_student_department(self):
        self.db.add(Student(
            name="Orphan", roll_number="ORPHAN", email="orphan@example.com", department_id=9999
        ))
        with self.assertRaises(IntegrityError):
            self.db.commit()
        self.db.rollback()

    def test_concurrent_duplicate_attendance_has_one_winner(self):
        with tempfile.TemporaryDirectory(prefix="smartattend-race-test-") as folder:
            race_engine = create_engine(
                f"sqlite:///{Path(folder) / 'race.sqlite'}",
                connect_args={"check_same_thread": False, "timeout": 15},
            )

            @event.listens_for(race_engine, "connect")
            def enable_race_foreign_keys(connection, _record):
                connection.execute("PRAGMA foreign_keys=ON")

            Base.metadata.create_all(race_engine)
            from sqlalchemy.orm import sessionmaker
            factory = sessionmaker(bind=race_engine)
            with factory.begin() as setup:
                department = Department(name="Concurrency department")
                setup.add(department)
                setup.flush()
                teacher = Teacher(name="Race teacher", email="race@example.com", department_id=department.id)
                student = Student(name="Race student", roll_number="R1", email="race-student@example.com", department_id=department.id)
                subject = Subject(name="Race subject", code="RACE1", department_id=department.id)
                setup.add_all([teacher, student, subject])
                setup.flush()
                session = AttendanceSession(
                    teacher_id=teacher.id, subject_id=subject.id,
                    session_date=date.today(), start_time=datetime.utcnow(), status="active",
                )
                setup.add(session)
                setup.flush()
                keys = (session.id, student.id)

            barrier = Barrier(2)

            def insert_record():
                db = factory()
                try:
                    barrier.wait(timeout=5)
                    db.add(AttendanceRecord(session_id=keys[0], student_id=keys[1], status="present"))
                    db.commit()
                    return "inserted"
                except IntegrityError:
                    db.rollback()
                    return "duplicate"
                finally:
                    db.close()

            with ThreadPoolExecutor(max_workers=2) as workers:
                outcomes = list(workers.map(lambda _index: insert_record(), range(2)))
            self.assertCountEqual(outcomes, ["inserted", "duplicate"])
            with factory() as verify:
                self.assertEqual(verify.query(AttendanceRecord).filter_by(session_id=keys[0]).count(), 1)
            race_engine.dispose()


if __name__ == "__main__":
    unittest.main()
