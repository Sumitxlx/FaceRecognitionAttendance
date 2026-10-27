import unittest
from datetime import date, datetime
import asyncio
import json
from unittest.mock import patch
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.main import (
    app,
    _visible_sessions,
    choose_face_match,
    create_attendance_session,
    end_attendance_session,
    get_attendance_records,
    get_dashboard,
)
from backend.main import decode_embedding_matrix
from backend.auth import hash_password, verify_password
from backend.database import get_db
from backend.models import (
    AttendanceAudit,
    AttendanceRecord,
    AttendanceSession,
    Base,
    Department,
    FaceEmbedding,
    Student,
    Subject,
    Teacher,
    User,
)
from backend.token import create_access_token


async def asgi_request(method, path, body=b"", headers=()):
    parsed = urlsplit(path)
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": parsed.path,
        "raw_path": parsed.path.encode(), "query_string": parsed.query.encode(),
        "root_path": "", "headers": list(headers),
        "client": ("testclient", 1234), "server": ("testserver", 80),
    }
    requests = [{"type": "http.request", "body": body, "more_body": False}]
    responses = []

    async def receive():
        if requests:
            return requests.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        responses.append(message)

    await app(scope, receive, send)
    status = next(item["status"] for item in responses if item["type"] == "http.response.start")
    response_body = b"".join(item.get("body", b"") for item in responses if item["type"] == "http.response.body")
    if not response_body:
        return status, None
    try:
        payload = json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = response_body.decode("utf-8", errors="replace")
    return status, payload


def one_image_multipart(path: Path, boundary: str) -> bytes:
    return b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="image"; filename="frame.jpg"\r\n',
        b"Content-Type: image/jpeg\r\n\r\n",
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])


class ApiAndMatchingTests(unittest.TestCase):
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
        self.department = Department(name="Computer science")
        self.other_department = Department(name="Physics")
        self.db.add_all([self.department, self.other_department])
        self.db.flush()
        self.teacher = Teacher(name="Teacher One", email="teacher@example.com", department_id=self.department.id)
        self.other_teacher = Teacher(name="Teacher Two", email="other@example.com", department_id=self.other_department.id)
        self.subject = Subject(name="Algorithms", code="ALG1", department_id=self.department.id)
        self.other_subject = Subject(name="Optics", code="OPT1", department_id=self.other_department.id)
        self.student = Student(name="Student One", roll_number="S1", email="student@example.com", department_id=self.department.id)
        self.other_student = Student(name="Student Two", roll_number="S2", email="student2@example.com", department_id=self.other_department.id)
        self.teacher_user = User(email=self.teacher.email, password_hash="unused", role="teacher")
        self.admin_user = User(email="admin@example.com", password_hash="unused", role="admin")
        self.student_user = User(email=self.student.email, password_hash="unused", role="student")
        self.db.add_all([self.teacher, self.other_teacher, self.subject, self.other_subject, self.student, self.other_student, self.teacher_user, self.admin_user, self.student_user])
        self.db.commit()
        self.db.query(User).filter(User.id == self.teacher_user.id).update({"password_hash": hash_password("integration passphrase")})
        self.db.commit()
        self.previous_override = app.dependency_overrides.get(get_db)
        app.dependency_overrides[get_db] = lambda: self.db

    def tearDown(self):
        if self.previous_override is None:
            app.dependency_overrides.pop(get_db, None)
        else:
            app.dependency_overrides[get_db] = self.previous_override
        self.db.close()
        self.engine.dispose()

    def test_teacher_session_lifecycle_and_scope(self):
        result = create_attendance_session({"subject_id": self.subject.id}, self.db, self.teacher_user)
        self.assertEqual(result["status"], "active")
        session_id = result["session_id"]
        self.assertEqual([item.id for item in _visible_sessions(self.db, self.teacher_user)], [session_id])

        with self.assertRaises(HTTPException) as denied:
            create_attendance_session({"subject_id": self.other_subject.id}, self.db, self.teacher_user)
        self.assertEqual(denied.exception.status_code, 403)

        ended = end_attendance_session(session_id, self.db, self.teacher_user)
        self.assertEqual(ended["status"], "completed")
        saved = self.db.get(AttendanceSession, session_id)
        self.assertIsNotNone(saved.end_time)
        with self.assertRaises(HTTPException) as repeated:
            end_attendance_session(session_id, self.db, self.teacher_user)
        self.assertEqual(repeated.exception.status_code, 409)

    def test_empty_session_can_be_cancelled_and_sessions_with_records_cannot(self):
        empty = create_attendance_session({"subject_id": self.subject.id}, self.db, self.teacher_user)
        admin_token = create_access_token({"sub": str(self.admin_user.id), "role": "admin"})
        admin_auth = [(b"authorization", f"Bearer {admin_token}".encode())]

        status, cancelled = asyncio.run(asgi_request(
            "POST", f"/api/attendance/sessions/{empty['session_id']}/cancel", headers=admin_auth
        ))
        self.assertEqual(status, 200, cancelled)
        self.assertEqual(cancelled["status"], "cancelled")
        saved_empty = self.db.get(AttendanceSession, empty["session_id"])
        self.assertEqual(saved_empty.status, "cancelled")
        self.assertIsNotNone(saved_empty.end_time)
        summary = get_dashboard(self.db, self.admin_user)
        self.assertEqual(summary["session_count"], 0)
        self.assertEqual(summary["completed_session_count"], 0)
        self.assertIsNone(summary["attendance_percentage"])

        recorded = create_attendance_session({"subject_id": self.subject.id}, self.db, self.teacher_user)
        self.db.add(AttendanceRecord(
            session_id=recorded["session_id"], student_id=self.student.id, status="present"
        ))
        self.db.commit()
        status, conflict = asyncio.run(asgi_request(
            "POST", f"/api/attendance/sessions/{recorded['session_id']}/cancel", headers=admin_auth
        ))
        self.assertEqual(status, 409, conflict)
        self.db.refresh(self.db.get(AttendanceSession, recorded["session_id"]))
        self.assertEqual(self.db.get(AttendanceSession, recorded["session_id"]).status, "active")

    def test_admin_academic_and_student_crud_persists_and_scopes(self):
        admin_token = create_access_token({"sub": str(self.admin_user.id), "role": "admin"})
        auth = [(b"authorization", f"Bearer {admin_token}".encode())]
        unauthenticated, _ = asyncio.run(asgi_request("GET", "/api/departments"))
        self.assertNotEqual(unauthenticated, 200)
        status, departments = asyncio.run(asgi_request("GET", "/api/departments", headers=auth))
        self.assertEqual(status, 200)
        self.assertEqual(len(departments), 2)

        status, department = asyncio.run(asgi_request("POST", "/api/departments",
            json.dumps({"name": "Electrical systems"}).encode(),
            auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 201, department)
        status, updated = asyncio.run(asgi_request("PUT", f"/api/departments/{department['id']}",
            json.dumps({"name": "Electrical engineering"}).encode(),
            auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 200)
        self.assertEqual(updated["name"], "Electrical engineering")

        status, teacher = asyncio.run(asgi_request("POST", "/api/teachers",
            json.dumps({"name": "New Teacher", "email": "new-teacher@example.com",
                        "department_id": department["id"]}).encode(),
            auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 201, teacher)
        status, account = asyncio.run(asgi_request("POST", "/api/register",
            json.dumps({"email": "new-teacher@example.com", "password": "teacher passphrase 123", "role": "teacher"}).encode(),
            auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 200, account)
        status, teacher_edit = asyncio.run(asgi_request("PUT", f"/api/teachers/{teacher['id']}",
            json.dumps({"name": "Updated Teacher", "email": "updated-teacher@example.com",
                        "department_id": department["id"]}).encode(),
            auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 200, teacher_edit)
        self.assertEqual(self.db.query(User).filter(User.id == account["id"]).one().email,
                         "updated-teacher@example.com")
        status, subject = asyncio.run(asgi_request("POST", "/api/subjects",
            json.dumps({"name": "Circuits", "code": "CIR101", "department_id": department["id"]}).encode(),
            auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 201, subject)
        status, subject_edit = asyncio.run(asgi_request("PUT", f"/api/subjects/{subject['id']}",
            json.dumps({"name": "Circuit Analysis", "code": "CIR101", "department_id": department["id"]}).encode(),
            auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 200, subject_edit)

        student_payload = {"name": "New Student", "roll_number": "N3", "email": "new@example.com",
                           "department_id": department["id"]}
        self.admin_user.username = "n3"
        self.db.commit()
        status, student = asyncio.run(asgi_request("POST", "/api/students", json.dumps(student_payload).encode(),
            auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 201, student)
        self.assertFalse(student["face_enrolled"])
        self.assertTrue(student["account_exists"])
        self.assertTrue(student["initial_credentials"]["temporary_password"])
        student_account = self.db.query(User).filter(User.student_id == student["id"]).one()
        self.assertTrue(student_account.must_change_password)
        self.assertNotEqual(student_account.username, "n3")
        self.assertEqual(self.db.query(User).filter(User.student_id == student["id"]).count(), 1)
        self.assertTrue(verify_password(student["initial_credentials"]["temporary_password"], student_account.password_hash))
        status, _ = asyncio.run(asgi_request("POST", "/api/students", json.dumps(student_payload).encode(),
            auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 409)
        status, login = asyncio.run(asgi_request("POST", "/api/login", json.dumps({
            "email": student_account.username,
            "password": student["initial_credentials"]["temporary_password"],
        }).encode(), [(b"content-type", b"application/json")]))
        self.assertEqual(status, 200, login)
        student_auth = [(b"authorization", f"Bearer {login['access_token']}".encode())]
        self.assertTrue(login["user"]["must_change_password"])
        status, blocked = asyncio.run(asgi_request("GET", "/api/dashboard", headers=student_auth))
        self.assertEqual(status, 403, blocked)
        status, changed = asyncio.run(asgi_request("POST", "/api/me/password", json.dumps({
            "current_password": student["initial_credentials"]["temporary_password"],
            "new_password": "new student password 123",
        }).encode(), student_auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 200, changed)
        self.assertFalse(student_account.must_change_password)
        status, personal_dashboard = asyncio.run(asgi_request("GET", "/api/dashboard", headers=student_auth))
        self.assertEqual(status, 200, personal_dashboard)
        self.assertEqual(personal_dashboard["student_attendance"][0]["student_id"], student["id"])
        student_payload["name"] = "Updated Student"
        student_payload["email"] = "updated-student@example.com"
        status, edited = asyncio.run(asgi_request("PUT", f"/api/students/{student['id']}",
            json.dumps(student_payload).encode(), auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 200, edited)
        self.assertEqual(edited["name"], "Updated Student")
        self.assertEqual(self.db.query(User).filter(User.id == student_account.id).one().email,
                         "updated-student@example.com")

    def test_admin_can_link_existing_student_once(self):
        admin_token = create_access_token({"sub": str(self.admin_user.id), "role": "admin"})
        admin_auth = [(b"authorization", f"Bearer {admin_token}".encode())]
        unlinked = Student(name="Unlinked", roll_number="LINK-1", email="unlinked@example.com",
                           department_id=self.department.id)
        self.db.add(unlinked)
        self.db.commit()
        status, credentials = asyncio.run(asgi_request(
            "POST", f"/api/students/{unlinked.id}/account", headers=admin_auth,
        ))
        self.assertEqual(status, 201, credentials)
        linked = self.db.query(User).filter(User.student_id == unlinked.id).one()
        self.assertTrue(linked.must_change_password)
        self.assertTrue(verify_password(credentials["temporary_password"], linked.password_hash))
        status, rotated = asyncio.run(asgi_request(
            "POST", f"/api/students/{unlinked.id}/account/reset-password", headers=admin_auth,
        ))
        self.assertEqual(status, 200, rotated)
        self.assertTrue(linked.must_change_password)
        self.assertFalse(verify_password(credentials["temporary_password"], linked.password_hash))
        self.assertTrue(verify_password(rotated["temporary_password"], linked.password_hash))
        status, duplicate = asyncio.run(asgi_request(
            "POST", f"/api/students/{unlinked.id}/account", headers=admin_auth,
        ))
        self.assertEqual(status, 409, duplicate)

    def test_class_setup_links_existing_teacher_and_subject_for_single_class(self):
        admin_token = create_access_token({"sub": str(self.admin_user.id), "role": "admin"})
        admin_auth = [(b"authorization", f"Bearer {admin_token}".encode())]
        teacher_profile = Teacher(name="Class Teacher", email="profile-contact@example.com",
                                  department_id=self.department.id)
        teacher_login = User(email="class-login@example.com", password_hash=hash_password("teacher password 123"), role="teacher")
        second_subject = Subject(name="Data structures", code="DS2", department_id=self.department.id)
        self.db.add_all([teacher_profile, teacher_login, second_subject])
        self.db.commit()
        teacher_token = create_access_token({"sub": str(teacher_login.id), "role": "teacher"})
        teacher_auth = [(b"authorization", f"Bearer {teacher_token}".encode())]
        status, _ = asyncio.run(asgi_request("GET", "/api/class-setup", headers=teacher_auth))
        self.assertEqual(status, 403)
        status, failure = asyncio.run(asgi_request("GET", "/api/subjects", headers=teacher_auth))
        self.assertEqual(status, 403)
        self.assertIn("Class setup", failure["detail"])

        status, options = asyncio.run(asgi_request("GET", "/api/class-setup", headers=admin_auth))
        self.assertEqual(status, 200, options)
        selected_department = next(item for item in options["departments"] if item["id"] == self.department.id)
        self.assertEqual(selected_department["student_count"], 1)
        status, configured = asyncio.run(asgi_request("POST", "/api/class-setup", json.dumps({
            "teacher_id": teacher_profile.id,
            "teacher_user_id": teacher_login.id,
            "subject_id": self.subject.id,
        }).encode(), admin_auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 200, configured)
        self.assertEqual(teacher_login.teacher_id, teacher_profile.id)
        self.assertEqual(teacher_login.email, "class-login@example.com")
        self.assertEqual(self.subject.teacher_id, teacher_profile.id)

        status, subjects = asyncio.run(asgi_request("GET", "/api/subjects", headers=teacher_auth))
        self.assertEqual(status, 200, subjects)
        self.assertEqual([row["id"] for row in subjects], [self.subject.id])
        status, denied_session = asyncio.run(asgi_request(
            "POST", "/api/attendance/sessions", json.dumps({"subject_id": second_subject.id}).encode(),
            teacher_auth + [(b"content-type", b"application/json")],
        ))
        self.assertEqual(status, 403, denied_session)
        status, other_teacher_subjects = asyncio.run(asgi_request(
            "GET", "/api/subjects",
            headers=[(b"authorization", f"Bearer {create_access_token({'sub': str(self.teacher_user.id), 'role': 'teacher'})}".encode())],
        ))
        self.assertEqual(status, 200, other_teacher_subjects)
        self.assertEqual([row["id"] for row in other_teacher_subjects], [second_subject.id])
        status, created = asyncio.run(asgi_request(
            "POST", "/api/attendance/sessions", json.dumps({"subject_id": self.subject.id}).encode(),
            teacher_auth + [(b"content-type", b"application/json")],
        ))
        self.assertEqual(status, 200, created)
        self.assertEqual(created["teacher_id"], teacher_profile.id)
        status, active_conflict = asyncio.run(asgi_request(
            "POST", "/api/attendance/sessions", json.dumps({"subject_id": self.subject.id}).encode(),
            teacher_auth + [(b"content-type", b"application/json")],
        ))
        self.assertEqual(status, 409, active_conflict)
        status, closed = asyncio.run(asgi_request(
            "POST", f"/api/attendance/sessions/{created['session_id']}/end", headers=teacher_auth,
        ))
        self.assertEqual(status, 200, closed)
        self.assertEqual(closed["status"], "completed")

        status, _ = asyncio.run(asgi_request("POST", "/api/class-setup", json.dumps({
            "teacher_id": teacher_profile.id,
            "teacher_user_id": teacher_login.id,
            "subject_id": second_subject.id,
        }).encode(), admin_auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 200)
        status, assigned_subjects = asyncio.run(asgi_request("GET", "/api/subjects", headers=teacher_auth))
        self.assertEqual(status, 200)
        self.assertEqual({row["id"] for row in assigned_subjects}, {self.subject.id, second_subject.id})

        status, repeated = asyncio.run(asgi_request("POST", "/api/class-setup", json.dumps({
            "teacher_id": teacher_profile.id,
            "teacher_user_id": teacher_login.id,
            "subject_id": self.subject.id,
        }).encode(), admin_auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 200, repeated)

    def test_student_and_temporary_account_creation_roll_back_together(self):
        admin_token = create_access_token({"sub": str(self.admin_user.id), "role": "admin"})
        admin_auth = [(b"authorization", f"Bearer {admin_token}".encode()), (b"content-type", b"application/json")]
        payload = {"name": "Atomic", "roll_number": "ATOMIC-1", "email": "atomic@example.com",
                   "department_id": self.department.id}
        with patch("backend.main.hash_password", side_effect=RuntimeError("simulated account failure")):
            status, _ = asyncio.run(asgi_request("POST", "/api/students", json.dumps(payload).encode(), admin_auth))
        self.assertEqual(status, 500)
        self.assertIsNone(self.db.query(Student).filter(Student.roll_number == "ATOMIC-1").first())
        self.assertIsNone(self.db.query(User).filter(User.email == "atomic@example.com").first())

        teacher_token = create_access_token({"sub": str(self.teacher_user.id), "role": "teacher"})
        teacher_auth = [(b"authorization", f"Bearer {teacher_token}".encode())]
        status, own_departments = asyncio.run(asgi_request("GET", "/api/departments", headers=teacher_auth))
        self.assertEqual(status, 200)
        self.assertEqual(len(own_departments), 1)
        status, _ = asyncio.run(asgi_request("POST", "/api/departments",
            json.dumps({"name": "Forbidden"}).encode(),
            teacher_auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 403)

    def test_manual_attendance_and_correction_are_audited(self):
        session = create_attendance_session({"subject_id": self.subject.id}, self.db, self.teacher_user)
        token = create_access_token({"sub": str(self.teacher_user.id), "role": "teacher"})
        auth = [(b"authorization", f"Bearer {token}".encode()), (b"content-type", b"application/json")]
        path = f"/api/attendance/sessions/{session['session_id']}/manual"
        status, entered = asyncio.run(asgi_request("POST", path,
            json.dumps({"student_id": self.student.id, "status": "present", "note": "Roll call"}).encode(), auth))
        self.assertEqual(status, 200, entered)
        self.assertEqual(entered["method"], "manual")
        status, corrected = asyncio.run(asgi_request("POST", path,
            json.dumps({"student_id": self.student.id, "status": "absent", "note": "Teacher correction"}).encode(), auth))
        self.assertEqual(status, 200, corrected)
        self.assertEqual(corrected["action"], "manual-correction")
        record = self.db.query(AttendanceRecord).filter_by(session_id=session["session_id"], student_id=self.student.id).one()
        self.assertEqual(record.status, "absent")
        self.assertEqual(record.recognition_method, "manual-correction")
        self.assertEqual(record.marked_by_user_id, self.teacher_user.id)
        audit = self.db.query(AttendanceAudit).filter_by(attendance_record_id=record.id).order_by(AttendanceAudit.id).all()
        self.assertEqual([item.action for item in audit], ["manual-entry", "manual-correction"])
        self.assertEqual(audit[-1].note, "Teacher correction")

    def test_profile_password_change_and_filtered_csv_export(self):
        token = create_access_token({"sub": str(self.teacher_user.id), "role": "teacher"})
        auth = [(b"authorization", f"Bearer {token}".encode())]
        status, profile = asyncio.run(asgi_request("GET", "/api/me", headers=auth))
        self.assertEqual(status, 200)
        self.assertEqual(profile["role"], "teacher")
        status, result = asyncio.run(asgi_request("POST", "/api/me/password",
            json.dumps({"current_password": "integration passphrase", "new_password": "new integration passphrase 1"}).encode(),
            auth + [(b"content-type", b"application/json")]))
        self.assertEqual(status, 200, result)
        self.assertTrue(verify_password("new integration passphrase 1", self.teacher_user.password_hash))

        session = AttendanceSession(teacher_id=self.teacher.id, subject_id=self.subject.id,
            session_date=date.today(), start_time=datetime.now(), status="completed", end_time=datetime.now())
        self.db.add(session)
        self.db.flush()
        self.db.add(AttendanceRecord(session_id=session.id, student_id=self.student.id, status="present"))
        self.db.commit()
        status, exported = asyncio.run(asgi_request("GET", "/api/attendance/records.csv?status=present", headers=auth))
        self.assertEqual(status, 200)
        self.assertIn("Student One", exported)
        self.assertIn("recognition_distance", exported)

    def test_analytics_uses_completed_sessions_and_student_scope(self):
        first = AttendanceSession(teacher_id=self.teacher.id, subject_id=self.subject.id,
            session_date=date.today(), start_time=datetime.now(), status="completed", end_time=datetime.now())
        second = AttendanceSession(teacher_id=self.teacher.id, subject_id=self.subject.id,
            session_date=date.today(), start_time=datetime.now(), status="completed", end_time=datetime.now())
        self.db.add_all([first, second])
        self.db.flush()
        classmate = Student(name="Classmate", roll_number="S3", email="classmate@example.com", department_id=self.department.id)
        self.db.add(classmate)
        self.db.flush()
        self.db.add_all([
            AttendanceRecord(session_id=first.id, student_id=self.student.id, status="present"),
            AttendanceRecord(session_id=first.id, student_id=classmate.id, status="present"),
        ])
        self.db.commit()
        admin_summary = get_dashboard(self.db, self.admin_user)
        self.assertEqual(admin_summary["attendance_percentage"], 50.0)
        self.assertEqual(admin_summary["attendance_denominator"], 4)
        self.assertEqual(admin_summary["defaulter_count"], 2)
        student_summary = get_dashboard(self.db, self.student_user)
        self.assertEqual(student_summary["student_count"], 1)
        self.assertEqual(student_summary["check_ins"], 1)
        self.assertEqual(student_summary["attendance_denominator"], 2)

    def test_records_and_dashboard_are_role_scoped_and_real(self):
        session = AttendanceSession(
            teacher_id=self.teacher.id,
            subject_id=self.subject.id,
            session_date=date.today(),
            start_time=datetime.now(),
            status="completed",
            end_time=datetime.now(),
        )
        hidden = AttendanceSession(
            teacher_id=self.other_teacher.id,
            subject_id=self.other_subject.id,
            session_date=date.today(),
            start_time=datetime.now(),
            status="completed",
            end_time=datetime.now(),
        )
        self.db.add_all([session, hidden])
        self.db.flush()
        self.db.add_all([
            AttendanceRecord(session_id=session.id, student_id=self.student.id, status="present"),
            AttendanceRecord(session_id=hidden.id, student_id=self.other_student.id, status="present"),
        ])
        self.db.commit()

        records = get_attendance_records(db=self.db, current_user=self.teacher_user)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["subject"], "Algorithms")
        dashboard = get_dashboard(self.db, self.teacher_user)
        self.assertEqual(dashboard["check_ins"], 1)
        self.assertEqual(dashboard["session_count"], 1)
        self.assertEqual(dashboard["student_count"], 1)

        own_records = get_attendance_records(db=self.db, current_user=self.student_user)
        self.assertEqual([item["student_id"] for item in own_records], [self.student.id])

    def test_stored_float32_embedding_matches_and_rejects_unknown_or_ambiguous(self):
        first = np.zeros(512, dtype=np.float32)
        first[0] = 1.0
        orthogonal = np.zeros(512, dtype=np.float32)
        orthogonal[1] = 1.0
        blob = np.stack([first, first]).astype(np.float32).tobytes()
        embedding = FaceEmbedding(student_id=self.student.id, embedding=blob)
        self.db.add(embedding)
        self.db.commit()

        stored = self.db.query(FaceEmbedding).filter_by(student_id=self.student.id).one()
        matched, distance, error = choose_face_match(first, [(self.student, stored.embedding)])
        self.assertEqual(matched.id, self.student.id)
        self.assertEqual(distance, 0.0)
        self.assertIsNone(error)

        unknown, _, reason = choose_face_match(orthogonal, [(self.student, stored.embedding)])
        self.assertIsNone(unknown)
        self.assertEqual(reason, "No confident match")

        ambiguous, _, reason = choose_face_match(first, [
            (self.student, stored.embedding),
            (self.other_student, stored.embedding),
        ])
        self.assertIsNone(ambiguous)
        self.assertEqual(reason, "Ambiguous face match")

    def test_http_api_login_session_write_and_close(self):
        status, login = asyncio.run(asgi_request(
            "POST", "/api/login",
            json.dumps({"email": self.teacher.email, "password": "integration passphrase"}).encode(),
            [(b"content-type", b"application/json")],
        ))
        self.assertEqual(status, 200, login)
        token = login["access_token"]
        auth = [(b"authorization", f"Bearer {token}".encode())]
        pages = ["/", "/login", "/setup", "/profile", "/teacher-dashboard", "/take-attendance", "/attendance-records", "/students", "/analytics", "/student-dashboard", "/admin-dashboard", "/class-setup"]
        for page in pages:
            page_status, html = asyncio.run(asgi_request("GET", page))
            self.assertEqual(page_status, 200, page)
            self.assertIn("SmartAttend", html, page)
        static_status, stylesheet = asyncio.run(asgi_request("GET", "/static/css/style.css"))
        self.assertEqual(static_status, 200)
        self.assertTrue(stylesheet)

        status, subjects = asyncio.run(asgi_request("GET", "/api/subjects", headers=auth))
        self.assertEqual(status, 200, subjects)
        self.assertEqual(subjects[0]["id"], self.subject.id)
        status, departments = asyncio.run(asgi_request("GET", "/api/departments", headers=auth))
        self.assertEqual(status, 200, departments)
        self.assertEqual(departments[0]["id"], self.department.id)
        status, visible_students = asyncio.run(asgi_request("GET", "/api/students", headers=auth))
        self.assertEqual(status, 200, visible_students)
        self.assertEqual({student["id"] for student in visible_students}, {self.student.id})
        status, _ = asyncio.run(asgi_request("GET", "/api/teachers", headers=auth))
        self.assertEqual(status, 403)
        status, _ = asyncio.run(asgi_request(
            "POST", "/api/register",
            json.dumps({"email": "blocked@example.com", "password": "long integration password", "role": "student"}).encode(),
            auth + [(b"content-type", b"application/json")],
        ))
        self.assertEqual(status, 403)
        admin_token = create_access_token({"sub": str(self.admin_user.id), "role": "admin"})
        admin_auth = [(b"authorization", f"Bearer {admin_token}".encode())]
        status, new_department = asyncio.run(asgi_request(
            "POST", "/api/departments", json.dumps({"name": "Mathematics"}).encode(),
            admin_auth + [(b"content-type", b"application/json")],
        ))
        self.assertEqual(status, 201, new_department)
        status, new_teacher = asyncio.run(asgi_request(
            "POST", "/api/teachers",
            json.dumps({"name": "Teacher Three", "email": "teacher3@example.com", "department_id": new_department["id"]}).encode(),
            admin_auth + [(b"content-type", b"application/json")],
        ))
        self.assertEqual(status, 201, new_teacher)
        status, new_subject = asyncio.run(asgi_request(
            "POST", "/api/subjects",
            json.dumps({"name": "Linear algebra", "code": "MATH101", "department_id": new_department["id"]}).encode(),
            admin_auth + [(b"content-type", b"application/json")],
        ))
        self.assertEqual(status, 201, new_subject)
        self.assertEqual(self.db.get(Teacher, new_teacher["id"]).department_id, new_department["id"])
        self.assertEqual(self.db.get(Subject, new_subject["id"]).department_id, new_department["id"])
        status, _ = asyncio.run(asgi_request(
            "POST", "/api/departments", json.dumps({"name": "mathematics"}).encode(),
            admin_auth + [(b"content-type", b"application/json")],
        ))
        self.assertEqual(status, 409)
        malformed = create_access_token({"sub": "not-an-id"})
        status, _ = asyncio.run(asgi_request(
            "GET", "/api/dashboard",
            headers=[(b"authorization", f"Bearer {malformed}".encode())],
        ))
        self.assertEqual(status, 401)

        status, created = asyncio.run(asgi_request(
            "POST", "/api/attendance/sessions",
            json.dumps({"subject_id": self.subject.id}).encode(),
            auth + [(b"content-type", b"application/json")],
        ))
        self.assertEqual(status, 200)
        persisted = self.db.get(AttendanceSession, created["session_id"])
        self.assertEqual(persisted.status, "active")
        self.assertIsNotNone(persisted.start_time)

        status, dashboard = asyncio.run(asgi_request("GET", "/api/dashboard", headers=auth))
        self.assertEqual(status, 200)
        self.assertEqual(dashboard["session_count"], 1)
        status, closed = asyncio.run(asgi_request(
            "POST", f"/api/attendance/sessions/{created['session_id']}/end", headers=auth
        ))
        self.assertEqual(status, 200)
        self.assertEqual(closed["status"], "completed")
        self.db.refresh(persisted)
        self.assertEqual(persisted.status, "completed")
        self.assertIsNotNone(persisted.end_time)

        self.db.add(Student(name="Account Student", roll_number="ACCT1", email="new-student@example.com",
                            department_id=self.department.id))
        self.db.commit()
        status, account = asyncio.run(asgi_request(
            "POST", "/api/register",
            json.dumps({"email": "new-student@example.com", "password": "long integration password", "role": "student"}).encode(),
            admin_auth + [(b"content-type", b"application/json")],
        ))
        self.assertEqual(status, 200, account)
        self.assertNotIn("password_hash", account)

    def test_real_face_enrollment_persists_embeddings_and_matches_readback(self):
        dataset = Path(__file__).resolve().parents[1] / "dataset" / "sumit"
        samples = sorted(dataset.glob("*.jpg"), key=lambda path: int(path.stem))
        self.assertEqual(len(samples), 30, "The checked-in real enrollment sample set changed")
        boundary = "smartattend-test-boundary"
        parts = []

        def field(name, value):
            parts.extend([
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode(), b"\r\n",
            ])

        field("student_id", self.student.id)
        field("department_id", self.department.id)
        for index, path in enumerate(samples, start=1):
            parts.extend([
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="images"; filename="sample-{index}.jpg"\r\n'.encode(),
                b"Content-Type: image/jpeg\r\n\r\n",
                path.read_bytes(), b"\r\n",
            ])
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)
        token = create_access_token({"sub": str(self.admin_user.id), "role": "admin"})
        status, response = asyncio.run(asgi_request(
            "POST", "/api/students/enroll-face", body,
            [
                (b"authorization", f"Bearer {token}".encode()),
                (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
            ],
        ))
        self.assertEqual(status, 200, response)
        self.assertEqual(response["student_id"], self.student.id)
        self.assertGreaterEqual(response["embeddings_processed"], 10)
        self.assertEqual(response["embeddings_processed"] + response["samples_rejected"], 30)

        saved_student = self.db.get(Student, response["student_id"])
        stored = self.db.query(FaceEmbedding).filter_by(student_id=saved_student.id).one()
        matrix = decode_embedding_matrix(stored.embedding)
        self.assertEqual(matrix.shape, (response["embeddings_processed"], 512))
        self.assertEqual(stored.model_name, "ArcFace")
        self.assertEqual(stored.vector_dimension, 512)
        self.assertNotEqual(stored.model_version, "legacy-unspecified")
        self.assertIn("FaceAligner", stored.preprocessing)
        matched, distance, error = choose_face_match(matrix[0], [(saved_student, stored.embedding)])
        self.assertEqual(matched.id, saved_student.id)
        self.assertAlmostEqual(distance, 0.0, places=5)
        self.assertIsNone(error)

        token = create_access_token({"sub": str(self.admin_user.id), "role": "admin"})
        auth = [(b"authorization", f"Bearer {token}".encode())]
        duplicate_status, duplicate_body = asyncio.run(asgi_request(
            "POST", "/api/students/enroll-face", body,
            auth + [(b"content-type", f"multipart/form-data; boundary={boundary}".encode())],
        ))
        self.assertEqual(duplicate_status, 409, duplicate_body)

        status, created = asyncio.run(asgi_request(
            "POST", "/api/attendance/sessions",
            json.dumps({"subject_id": self.subject.id, "teacher_id": self.teacher.id}).encode(),
            auth + [(b"content-type", b"application/json")],
        ))
        self.assertEqual(status, 200, created)
        session_id = created["session_id"]

        image_boundary = "smartattend-frame-boundary"
        image_headers = auth + [(b"content-type", f"multipart/form-data; boundary={image_boundary}".encode())]
        root = Path(__file__).resolve().parents[1]
        unknown_path = root / "dataset" / "himanshu" / "8.jpg"
        status, unknown_result = asyncio.run(asgi_request(
            "POST", f"/api/attendance/sessions/{session_id}/recognize",
            one_image_multipart(unknown_path, image_boundary), image_headers,
        ))
        self.assertEqual(status, 200, unknown_result)
        self.assertFalse(unknown_result["recognized"])
        self.assertEqual(self.db.query(AttendanceRecord).filter_by(session_id=session_id).count(), 0)

        genuine_path = dataset / "8.jpg"
        status, recognized = asyncio.run(asgi_request(
            "POST", f"/api/attendance/sessions/{session_id}/recognize",
            one_image_multipart(genuine_path, image_boundary), image_headers,
        ))
        self.assertEqual(status, 200, recognized)
        self.assertTrue(recognized["recognized"])
        self.assertFalse(recognized["already_marked"])
        self.assertEqual(recognized["student_id"], saved_student.id)

        status, duplicate = asyncio.run(asgi_request(
            "POST", f"/api/attendance/sessions/{session_id}/recognize",
            one_image_multipart(genuine_path, image_boundary), image_headers,
        ))
        self.assertEqual(status, 200, duplicate)
        self.assertTrue(duplicate["already_marked"])
        self.assertEqual(self.db.query(AttendanceRecord).filter_by(session_id=session_id).count(), 1)

        status, closed = asyncio.run(asgi_request(
            "POST", f"/api/attendance/sessions/{session_id}/end", headers=auth
        ))
        self.assertEqual(status, 200, closed)
        self.assertEqual(self.db.get(AttendanceSession, session_id).status, "completed")

        status, rows = asyncio.run(asgi_request(
            "GET", f"/api/attendance/records?session_id={session_id}", headers=auth
        ))
        self.assertEqual(status, 200)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["method"], "face")
        self.assertIsNotNone(rows[0]["recognition_distance"])
        status, dashboard = asyncio.run(asgi_request("GET", "/api/dashboard", headers=auth))
        self.assertEqual(status, 200)
        self.assertEqual(dashboard["check_ins"], 1)
        self.assertEqual(dashboard["completed_session_count"], 1)
        self.assertEqual(dashboard["attendance_percentage"], 100.0)


    def test_invalid_new_student_enrollment_leaves_no_partial_rows(self):
        boundary = "smartattend-invalid-enrollment"
        parts = []
        for name, value in [
            ("name", "Invalid enrollment test"),
            ("roll_number", "BAD-IMAGE-1"),
            ("email", "bad-image@example.com"),
            ("department_id", self.department.id),
        ]:
            parts.extend([
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode(), b"\r\n",
            ])
        for index in range(30):
            parts.extend([
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="images"; filename="invalid-{index}.jpg"\r\n'.encode(),
                b"Content-Type: image/jpeg\r\n\r\nnot-an-image\r\n",
            ])
        parts.append(f"--{boundary}--\r\n".encode())
        token = create_access_token({"sub": str(self.admin_user.id), "role": "admin"})
        status, response = asyncio.run(asgi_request(
            "POST", "/api/students/enroll-face", b"".join(parts),
            [
                (b"authorization", f"Bearer {token}".encode()),
                (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
            ],
        ))
        self.assertEqual(status, 422, response)
        self.assertIsNone(self.db.query(Student).filter_by(roll_number="BAD-IMAGE-1").first())
        self.assertEqual(self.db.query(FaceEmbedding).count(), 0)

class FirstAdminApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        self.previous_override = app.dependency_overrides.get(get_db)
        app.dependency_overrides[get_db] = lambda: self.db

    def tearDown(self):
        if self.previous_override is None:
            app.dependency_overrides.pop(get_db, None)
        else:
            app.dependency_overrides[get_db] = self.previous_override
        self.db.close()
        self.engine.dispose()

    def test_initial_admin_api_is_one_time_and_returns_authenticated_user(self):
        status, state = asyncio.run(asgi_request("GET", "/api/setup-status"))
        self.assertEqual(status, 200)
        self.assertTrue(state["setup_required"])

        status, result = asyncio.run(asgi_request(
            "POST", "/api/setup/initial-admin",
            json.dumps({"email": "first-admin@example.com", "password": "long setup password"}).encode(),
            [(b"content-type", b"application/json")],
        ))
        self.assertEqual(status, 201, result)
        self.assertEqual(result["user"]["role"], "admin")
        self.assertNotIn("password_hash", result["user"])
        user = self.db.query(User).filter_by(email="first-admin@example.com").one()
        self.assertTrue(verify_password("long setup password", user.password_hash))

        auth = [(b"authorization", f"Bearer {result['access_token']}".encode())]
        status, identity = asyncio.run(asgi_request("GET", "/api/me", headers=auth))
        self.assertEqual(status, 200)
        self.assertEqual(identity["email"], user.email)
        status, state = asyncio.run(asgi_request("GET", "/api/setup-status"))
        self.assertFalse(state["setup_required"])
        status, error = asyncio.run(asgi_request(
            "POST", "/api/setup/initial-admin",
            json.dumps({"email": "second-admin@example.com", "password": "another setup pass"}).encode(),
            [(b"content-type", b"application/json")],
        ))
        self.assertEqual(status, 409, error)



if __name__ == "__main__":
    unittest.main()
