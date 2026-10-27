
import csv
import cv2
import io
import numpy as np
import os
import re
import secrets
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.models import (
    AttendanceAudit,
    AttendanceRecord,
    AttendanceSession,
    Department,
    FaceEmbedding,
    Student,
    Subject,
    Teacher,
    User,
)
from backend.ai.recognition import (
    EMBEDDING_DTYPE,
    MODEL_NAME,
    MODEL_VERSION,
    MAX_COSINE_DISTANCE,
    MIN_MATCH_MARGIN,
    PREPROCESSING_METADATA,
    VECTOR_DIMENSION,
    FaceRecognition,
    choose_face_match,
    decode_embedding_matrix,
)
from backend.attendance_service import (
    AttendanceConflictError,
    SessionNotActiveError,
    StudentNotEligibleError,
    record_check_in,
    record_manual_attendance,
)

from backend.auth import hash_password, verify_password
from backend.database import get_db
from backend.database import engine
from backend.dependencies import get_current_user, require_role
from backend.migrations import upgrade_database
from backend.schemas import (
    DepartmentCreate,
    DepartmentUpdate,
    ChangePasswordRequest,
    FirstAdminRequest,
    LoginRequest,
    ManualAttendanceRequest,
    RegisterRequest,
    StudentCreate,
    StudentUpdate,
    SubjectCreate,
    SubjectUpdate,
    SingleClassSetupRequest,
    TeacherCreate,
    TeacherUpdate,
    UserResponse,
)
from backend.token import create_access_token
from backend.time_utils import utc_now_naive

ATTENDANCE_THRESHOLD_PERCENT = float(os.getenv("SMARTATTEND_ATTENDANCE_THRESHOLD_PERCENT", "75"))
if not 0 <= ATTENDANCE_THRESHOLD_PERCENT <= 100:
    raise RuntimeError("SMARTATTEND_ATTENDANCE_THRESHOLD_PERCENT must be between 0 and 100")


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"


# =========================================================
# FASTAPI APP
# =========================================================

@asynccontextmanager
async def lifespan(_app):
    upgrade_database(engine)
    yield


app = FastAPI(
    title="SmartAttend",
    description="Privacy-first attendance management system",
    version="1.0.0",
    lifespan=lifespan,
)

# Keep the app's existing frontend structure.
app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "application": "SmartAttend",
        "message": "FastAPI backend is running",
    }


@app.get("/api/setup-status")
def setup_status(db: Session = Depends(get_db)):
    admin_exists = db.query(User.id).filter(User.role == "admin").first() is not None
    return {"setup_required": not admin_exists}


@app.get("/api/me")
def current_user_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    profile = {"id": current_user.id, "email": current_user.email, "role": current_user.role,
               "must_change_password": current_user.must_change_password,
               "username": current_user.username}
    if current_user.role == "student":
        student = _student_for_user(db, current_user)
        profile.update({"name": student.name, "roll_number": student.roll_number,
                        "department": student.department.name})
    return profile


@app.post("/api/me/password")
def change_password(
    payload: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if len(payload.new_password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="New password must be no more than 72 UTF-8 bytes")
    if not verify_password(payload.current_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    current_user.password_hash = hash_password(payload.new_password)
    current_user.must_change_password = False
    db.commit()
    return {"saved": True, "message": "Password updated. Your account is ready to use."}


@app.post("/api/setup/initial-admin", status_code=201)
def create_initial_admin(
    request: Request,
    payload: FirstAdminRequest,
    db: Session = Depends(get_db),
):
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=403, detail="First-admin setup is available only from localhost")
    try:
        # BEGIN IMMEDIATE serializes concurrent first-admin setup attempts.
        db.execute(text("BEGIN IMMEDIATE"))
        if db.query(User.id).filter(User.role == "admin").first():
            raise HTTPException(status_code=409, detail="Administrator setup is already complete")
        email = str(payload.email).strip().lower()
        if db.query(User.id).filter(func.lower(User.email) == email).first():
            raise HTTPException(status_code=409, detail="That email already has an account")
        if len(payload.password.encode("utf-8")) > 72:
            raise HTTPException(status_code=400, detail="Password must be no more than 72 UTF-8 bytes")
        user = User(email=email, password_hash=hash_password(payload.password), role="admin")
        db.add(user)
        db.commit()
        return {
            "access_token": create_access_token({"sub": str(user.id), "role": user.role}),
            "token_type": "bearer",
            "user": {"id": user.id, "email": user.email, "role": user.role},
        }
    except HTTPException:
        db.rollback()
        raise
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Administrator setup was completed concurrently")
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not create the first administrator")


# =========================================================
# HTML PAGE ROUTES
# =========================================================

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"active_page": "home"},
    )


@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={},
    )


@app.get("/setup")
def setup_page(request: Request):
    return templates.TemplateResponse(request=request, name="setup.html", context={})


@app.get("/profile")
def profile_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="profile.html",
        context={"active_page": "profile", "allowed_roles": "admin,teacher,student"},
    )


@app.get("/teacher-dashboard")
def teacher_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboards/teacher-dashboard.html",
        context={"active_page": "dashboard", "allowed_roles": "admin,teacher"},
    )


@app.get("/class-setup")
def class_setup_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="class-setup.html",
        context={"active_page": "class-setup", "allowed_roles": "admin"},
    )


@app.get("/take-attendance")
def take_attendance(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="take-attendance.html",
        context={"active_page": "take-attendance", "allowed_roles": "admin,teacher"},
    )


@app.get("/attendance-records")
def attendance_records(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="attendance-records.html",
        context={"active_page": "records", "allowed_roles": "admin,teacher,student"},
    )


@app.get("/students")
def students_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="students.html",
        context={"active_page": "students", "allowed_roles": "admin,teacher"},
    )


@app.get("/analytics")
def analytics_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="analytics.html",
        context={"active_page": "analytics", "allowed_roles": "admin,teacher,student"},
    )


@app.get("/student-dashboard")
def student_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="student-dashboard.html",
        context={"active_page": "student-dashboard", "allowed_roles": "student"},
    )


@app.get("/admin-dashboard")
def admin_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="admin-dashboard.html",
        context={"active_page": "admin-dashboard", "allowed_roles": "admin"},
    )


# =========================================================
# STUDENTS API
# =========================================================

@app.get("/api/students")
def get_students(
    search: str | None = None,
    department_id: int | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role == "teacher":
        teacher = _teacher_for_user(db, current_user)
        query = db.query(Student).filter(Student.department_id == teacher.department_id)
        if department_id is not None and department_id != teacher.department_id:
            raise HTTPException(status_code=403, detail="You cannot view another department")
    elif current_user.role == "admin":
        query = db.query(Student)
    else:
        raise HTTPException(status_code=403, detail="You are not authorized to view the student directory")

    if department_id is not None:
        query = query.filter(Student.department_id == department_id)
    if search and search.strip():
        needle = f"%{search.strip()}%"
        query = query.filter(
            (Student.name.ilike(needle))
            | (Student.email.ilike(needle))
            | (Student.roll_number.ilike(needle))
        )
    return [
        {
            "id": student.id,
            "name": student.name,
            "roll_number": student.roll_number,
            "email": student.email,
            "department_id": student.department_id,
            "department_name": student.department.name,
            "face_enrolled": embedding_is_current(student.face_embedding),
            "has_embedding": student.face_embedding is not None,
            "embedding_count": (
                len(decode_embedding_matrix(student.face_embedding.embedding))
                if embedding_is_current(student.face_embedding) else 0
            ),
            "account_exists": bool(db.query(User.id).filter(
                User.role == "student",
                ((User.student_id == student.id) | (func.lower(User.email) == student.email.lower()))
            ).first()),
        }
        for student in query.order_by(Student.name).all()
    ]


@app.post("/api/students", status_code=201)
def create_student(
    payload: StudentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role == "teacher":
        teacher = _teacher_for_user(db, current_user)
        if payload.department_id != teacher.department_id:
            raise HTTPException(status_code=403, detail="You can add students only in your department")
    elif current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Staff access required")
    if db.get(Department, payload.department_id) is None:
        raise HTTPException(status_code=404, detail="Department not found")
    name, roll, email = payload.name.strip(), payload.roll_number.strip(), str(payload.email).strip().lower()
    if not name or not roll:
        raise HTTPException(status_code=400, detail="Student name and roll number are required")
    if db.query(Student).filter(
        (func.lower(Student.email) == email) | (func.lower(Student.roll_number) == roll.lower())
    ).first():
        raise HTTPException(status_code=409, detail="Student email or roll number already exists")
    if db.query(User).filter(func.lower(User.email) == email).first():
        raise HTTPException(status_code=409, detail="Student email is already used by another account")
    student = Student(name=name, roll_number=roll, email=email, department_id=payload.department_id)
    db.add(student)
    credentials = None
    try:
        db.flush()
        if current_user.role == "admin":
            username = _unique_student_username(db, roll)
            temporary_password = secrets.token_urlsafe(18)
            student_user = User(email=email, username=username, student_id=student.id,
                                password_hash=hash_password(temporary_password), role="student",
                                must_change_password=True)
            db.add(student_user)
            db.flush()
            credentials = {"username": username, "email": email, "temporary_password": temporary_password}
        db.commit()
        db.refresh(student)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Student email or roll number already exists")
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Student and login account could not be saved")
    return {"id": student.id, "name": student.name, "roll_number": student.roll_number,
            "email": student.email, "department_id": student.department_id,
            "face_enrolled": False, "embedding_count": 0, "account_exists": credentials is not None,
            "initial_credentials": credentials}


def _unique_student_username(db: Session, roll_number: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "", roll_number.lower())[:40] or "student"
    candidate = base
    while db.query(User.id).filter(func.lower(User.username) == candidate.lower()).first():
        candidate = f"{base[:32]}{secrets.token_hex(3)}"
    return candidate


@app.post("/api/students/{student_id}/account", status_code=201)
def link_student_account(
    student_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    student = db.get(Student, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    linked = db.query(User).filter(
        User.role == "student",
        ((User.student_id == student.id) | (func.lower(User.email) == student.email.lower()))
    ).first()
    if linked:
        if linked.student_id is None:
            linked.student_id = student.id
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                raise HTTPException(status_code=409, detail="This account is linked to another student")
        raise HTTPException(status_code=409, detail="This student already has a login account")
    username = _unique_student_username(db, student.roll_number)
    temporary_password = secrets.token_urlsafe(18)
    user = User(email=student.email, username=username, student_id=student.id,
                password_hash=hash_password(temporary_password), role="student", must_change_password=True)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="A login account already exists for this student")
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not create the student login account")
    return {"student_id": student.id, "username": username, "email": student.email,
            "temporary_password": temporary_password}


@app.post("/api/students/{student_id}/account/reset-password")
def reset_student_account_password(
    student_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    student = db.get(Student, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    user = db.query(User).filter(
        User.role == "student",
        ((User.student_id == student.id) | (func.lower(User.email) == student.email.lower()))
    ).first()
    if user is None:
        raise HTTPException(status_code=404, detail="This student does not have a login account")
    username = user.username or _unique_student_username(db, student.roll_number)
    temporary_password = secrets.token_urlsafe(18)
    user.username = username
    user.student_id = student.id
    user.password_hash = hash_password(temporary_password)
    user.must_change_password = True
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Could not reset this student account")
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not reset this student account")
    return {"student_id": student.id, "username": username, "email": user.email,
            "temporary_password": temporary_password}


@app.put("/api/students/{student_id}")
def update_student(
    student_id: int,
    payload: StudentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    student = db.get(Student, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    if current_user.role == "teacher":
        teacher = _teacher_for_user(db, current_user)
        if student.department_id != teacher.department_id or payload.department_id != teacher.department_id:
            raise HTTPException(status_code=403, detail="You can edit students only in your department")
    elif current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Staff access required")
    if db.get(Department, payload.department_id) is None:
        raise HTTPException(status_code=404, detail="Department not found")
    name, roll, email = payload.name.strip(), payload.roll_number.strip(), str(payload.email).strip().lower()
    if not name or not roll:
        raise HTTPException(status_code=400, detail="Student name and roll number are required")
    duplicate = db.query(Student).filter(
        Student.id != student_id,
        (func.lower(Student.email) == email) | (func.lower(Student.roll_number) == roll.lower()),
    ).first()
    if duplicate:
        raise HTTPException(status_code=409, detail="Student email or roll number already exists")
    if student.department_id != payload.department_id and (
        student.face_embedding is not None
        or db.query(AttendanceRecord.id).filter(AttendanceRecord.student_id == student_id).first()
    ):
        raise HTTPException(status_code=409, detail="A student with face or attendance history cannot be moved to another department")
    linked_user = db.query(User).filter(User.student_id == student.id, User.role == "student").first()
    if linked_user is None:
        linked_user = db.query(User).filter(func.lower(User.email) == student.email.lower(), User.role == "student").first()
    email_owner = db.query(User).filter(func.lower(User.email) == email).first()
    if email_owner and (linked_user is None or email_owner.id != linked_user.id):
        raise HTTPException(status_code=409, detail="New email is already used by another account")
    if linked_user and linked_user.email.lower() != email:
        linked_user.email = email
    if linked_user:
        linked_user.student_id = student.id
    student.name, student.roll_number, student.email = name, roll, email
    student.department_id = payload.department_id
    db.commit()
    return {"id": student.id, "name": student.name, "roll_number": student.roll_number,
            "email": student.email, "department_id": student.department_id,
            "face_enrolled": embedding_is_current(student.face_embedding),
            "embedding_count": (len(decode_embedding_matrix(student.face_embedding.embedding))
                                if embedding_is_current(student.face_embedding) else 0)}


def embedding_is_current(record: FaceEmbedding | None) -> bool:
    return bool(
        record
        and record.model_name == MODEL_NAME
        and record.model_version == MODEL_VERSION
        and record.vector_dimension == VECTOR_DIMENSION
        and record.preprocessing == PREPROCESSING_METADATA
        and decode_embedding_matrix(record.embedding) is not None
    )


def _teacher_for_user(db: Session, user: User) -> Teacher:
    teacher = db.get(Teacher, user.teacher_id) if user.teacher_id else None
    if teacher is None and user.teacher_id is None:
        teacher = db.query(Teacher).filter(func.lower(Teacher.email) == user.email.lower()).first()
    if teacher is None:
        raise HTTPException(
            status_code=403,
            detail=(
                "This teacher login is not linked to a teacher profile. Ask an administrator to open "
                "Class setup and link your account to the correct teacher profile."
            ),
        )
    return teacher


def _student_for_user(db: Session, user: User) -> Student:
    student = db.get(Student, user.student_id) if user.student_id else None
    if student is None:
        student = db.query(Student).filter(func.lower(Student.email) == user.email.lower()).first()
    if student is None:
        raise HTTPException(status_code=403, detail="Your user account is not linked to a student record")
    return student


def _visible_sessions(db: Session, user: User) -> list[AttendanceSession]:
    query = db.query(AttendanceSession)
    if user.role == "teacher":
        teacher = _teacher_for_user(db, user)
        query = query.filter(AttendanceSession.teacher_id == teacher.id)
    elif user.role == "student":
        student = _student_for_user(db, user)
        query = query.join(AttendanceRecord).filter(AttendanceRecord.student_id == student.id)
    elif user.role != "admin":
        raise HTTPException(status_code=403, detail="Staff access required")
    return query.order_by(AttendanceSession.start_time.desc()).all()


@app.get("/api/departments")
def get_departments(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role == "admin":
        departments = db.query(Department).order_by(Department.name).all()
    elif current_user.role == "teacher":
        teacher = _teacher_for_user(db, current_user)
        departments = db.query(Department).filter(Department.id == teacher.department_id).all()
    else:
        raise HTTPException(status_code=403, detail="Staff access required")
    return [{"id": item.id, "name": item.name} for item in departments]


@app.post("/api/departments", status_code=201)
def create_department(
    payload: DepartmentCreate,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_role("admin")),
):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Department name is required")
    if db.query(Department).filter(func.lower(Department.name) == name.lower()).first():
        raise HTTPException(status_code=409, detail="Department already exists")
    department = Department(name=name)
    db.add(department)
    try:
        db.commit()
        db.refresh(department)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Department already exists")
    return {"id": department.id, "name": department.name}


@app.put("/api/departments/{department_id}")
def update_department(
    department_id: int,
    payload: DepartmentUpdate,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_role("admin")),
):
    department = db.get(Department, department_id)
    if department is None:
        raise HTTPException(status_code=404, detail="Department not found")
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Department name is required")
    duplicate = db.query(Department).filter(
        func.lower(Department.name) == name.lower(), Department.id != department_id
    ).first()
    if duplicate:
        raise HTTPException(status_code=409, detail="Department already exists")
    department.name = name
    db.commit()
    return {"id": department.id, "name": department.name}


@app.get("/api/subjects")
def get_subjects(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(Subject)
    if current_user.role == "teacher":
        teacher = _teacher_for_user(db, current_user)
        assigned = db.query(Subject.id).filter(
            Subject.department_id == teacher.department_id,
            Subject.teacher_id == teacher.id,
        ).first()
        query = query.filter(Subject.department_id == teacher.department_id)
        query = query.filter(Subject.teacher_id == teacher.id if assigned else Subject.teacher_id.is_(None))
    elif current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Staff access required")
    return [
        {"id": item.id, "name": item.name, "code": item.code, "department_id": item.department_id,
         "department_name": item.department.name, "teacher_id": item.teacher_id}
        for item in query.order_by(Subject.name).all()
    ]


@app.get("/api/class-setup")
def get_class_setup(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_role("admin")),
):
    departments = db.query(Department).order_by(Department.name).all()
    teachers = db.query(Teacher).order_by(Teacher.name).all()
    subjects = db.query(Subject).order_by(Subject.name).all()
    users = db.query(User).filter(User.role == "teacher").order_by(User.email).all()
    department_rows = []
    for department in departments:
        students = db.query(Student).filter(Student.department_id == department.id).all()
        student_ids = [student.id for student in students]
        student_account_count = db.query(User.id).filter(
            User.role == "student", User.student_id.in_(student_ids) if student_ids else False
        ).count()
        session_count = db.query(AttendanceSession.id).join(Subject).filter(
            Subject.department_id == department.id
        ).count()
        active_session_count = db.query(AttendanceSession.id).join(Subject).filter(
            Subject.department_id == department.id,
            AttendanceSession.status == "active",
        ).count()
        department_rows.append({
            "id": department.id,
            "name": department.name,
            "student_count": len(students),
            "student_account_count": student_account_count,
            "face_enrolled_count": sum(embedding_is_current(student.face_embedding) for student in students),
            "teacher_count": sum(item.department_id == department.id for item in teachers),
            "subject_count": sum(item.department_id == department.id for item in subjects),
            "session_count": session_count,
            "active_session_count": active_session_count,
        })
    return {
        "departments": department_rows,
        "teachers": [{"id": item.id, "name": item.name, "email": item.email,
                      "department_id": item.department_id,
                      "account_linked": bool(db.query(User.id).filter(
                          User.role == "teacher",
                          ((User.teacher_id == item.id) | ((User.teacher_id.is_(None)) & (func.lower(User.email) == item.email.lower())))
                      ).first()), "subject_ids": [subject.id for subject in item.subjects]}
                     for item in teachers],
        "subjects": [{"id": item.id, "name": item.name, "code": item.code,
                      "department_id": item.department_id, "teacher_id": item.teacher_id}
                     for item in subjects],
        "teacher_accounts": [{"id": item.id, "email": item.email,
                              "teacher_id": item.teacher_id if item.teacher_id is not None else (
                                  db.query(Teacher.id).filter(func.lower(Teacher.email) == item.email.lower()).scalar()
                              )}
                             for item in users],
    }


@app.post("/api/class-setup")
def set_up_single_class(
    payload: SingleClassSetupRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_role("admin")),
):
    teacher = db.get(Teacher, payload.teacher_id)
    account = db.get(User, payload.teacher_user_id)
    subject = db.get(Subject, payload.subject_id)
    if teacher is None or subject is None:
        raise HTTPException(status_code=404, detail="Select an existing teacher and subject")
    if account is None or account.role != "teacher":
        raise HTTPException(status_code=404, detail="Select an existing teacher login")
    if teacher.department_id != subject.department_id:
        raise HTTPException(status_code=400, detail="The teacher and subject must belong to the same department")
    student_count = db.query(Student.id).filter(Student.department_id == teacher.department_id).count()
    if student_count == 0:
        raise HTTPException(status_code=409, detail="This department has no students. Add a real student before setting up the class")
    account_teacher = db.get(Teacher, account.teacher_id) if account.teacher_id else None
    if account_teacher is None and account.teacher_id is None:
        account_teacher = db.query(Teacher).filter(func.lower(Teacher.email) == account.email.lower()).first()
    if account_teacher is not None and account_teacher.id != teacher.id:
        raise HTTPException(status_code=409, detail="That login is already linked to a different teacher profile")
    teacher_account = db.query(User).filter(
        User.role == "teacher",
        ((User.teacher_id == teacher.id) | ((User.teacher_id.is_(None)) & (func.lower(User.email) == teacher.email.lower())))
    ).first()
    if teacher_account is not None and teacher_account.id != account.id:
        raise HTTPException(status_code=409, detail="This teacher profile is already linked to another login")
    if subject.teacher_id is not None and subject.teacher_id != teacher.id:
        raise HTTPException(status_code=409, detail="This subject is assigned to another teacher")
    if account.teacher_id is not None and account.teacher_id != teacher.id:
        raise HTTPException(status_code=409, detail="That teacher login is already assigned to another profile")
    try:
        account.teacher_id = teacher.id
        subject.teacher_id = teacher.id
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="The class assignment changed concurrently; reload setup")
    return {
        "saved": True,
        "department": teacher.department.name,
        "teacher": teacher.name,
        "teacher_email": account.email,
        "subject": subject.name,
        "subject_code": subject.code,
        "student_count": student_count,
        "face_enrolled_count": sum(embedding_is_current(student.face_embedding) for student in db.query(Student).filter(
            Student.department_id == teacher.department_id
        ).all()),
        "message": "Class is ready. The teacher will see this subject automatically when it is their only assigned class subject.",
    }


@app.post("/api/subjects", status_code=201)
def create_subject(
    payload: SubjectCreate,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_role("admin")),
):
    name, code = payload.name.strip(), payload.code.strip().upper()
    department = db.query(Department).filter(Department.id == payload.department_id).first()
    if department is None:
        raise HTTPException(status_code=404, detail="Department not found")
    if not name or not code:
        raise HTTPException(status_code=400, detail="Subject name and code are required")
    if db.query(Subject).filter(func.lower(Subject.code) == code.lower()).first():
        raise HTTPException(status_code=409, detail="Subject code already exists")
    subject = Subject(name=name, code=code, department_id=department.id)
    db.add(subject)
    try:
        db.commit()
        db.refresh(subject)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Subject code already exists")
    return {"id": subject.id, "name": subject.name, "code": subject.code, "department_id": subject.department_id}


@app.put("/api/subjects/{subject_id}")
def update_subject(
    subject_id: int,
    payload: SubjectUpdate,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_role("admin")),
):
    subject = db.get(Subject, subject_id)
    if subject is None:
        raise HTTPException(status_code=404, detail="Subject not found")
    department = db.get(Department, payload.department_id)
    if department is None:
        raise HTTPException(status_code=404, detail="Department not found")
    name, code = payload.name.strip(), payload.code.strip().upper()
    duplicate = db.query(Subject).filter(
        func.lower(Subject.code) == code.lower(), Subject.id != subject_id
    ).first()
    if not name or not code:
        raise HTTPException(status_code=400, detail="Subject name and code are required")
    if duplicate:
        raise HTTPException(status_code=409, detail="Subject code already exists")
    if subject.department_id != department.id and db.query(AttendanceSession.id).filter(
        AttendanceSession.subject_id == subject_id
    ).first():
        raise HTTPException(status_code=409, detail="A subject with attendance history cannot be moved to another department")
    if subject.teacher_id is not None:
        assigned_teacher = db.get(Teacher, subject.teacher_id)
        if assigned_teacher is None or assigned_teacher.department_id != department.id:
            raise HTTPException(status_code=409, detail="An assigned subject cannot be moved outside its teacher's department")
    subject.name, subject.code, subject.department_id = name, code, department.id
    db.commit()
    return {"id": subject.id, "name": subject.name, "code": subject.code, "department_id": subject.department_id}


@app.get("/api/teachers")
def get_teachers(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    return [
        {"id": item.id, "name": item.name, "email": item.email, "department_id": item.department_id,
         "account_linked": bool(db.query(User.id).filter(
             User.role == "teacher",
             ((User.teacher_id == item.id) | ((User.teacher_id.is_(None)) & (func.lower(User.email) == item.email.lower())))
         ).first())}
        for item in db.query(Teacher).order_by(Teacher.name).all()
    ]


@app.post("/api/teachers", status_code=201)
def create_teacher(
    payload: TeacherCreate,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_role("admin")),
):
    name = payload.name.strip()
    email = str(payload.email).strip().lower()
    department = db.query(Department).filter(Department.id == payload.department_id).first()
    if department is None:
        raise HTTPException(status_code=404, detail="Department not found")
    if not name:
        raise HTTPException(status_code=400, detail="Teacher name is required")
    if db.query(Teacher).filter(func.lower(Teacher.email) == email).first():
        raise HTTPException(status_code=409, detail="Teacher email already exists")
    if db.query(User).filter(func.lower(User.email) == email).first():
        raise HTTPException(status_code=409, detail="Teacher email is already used by another account")
    teacher = Teacher(name=name, email=email, department_id=department.id)
    db.add(teacher)
    try:
        db.commit()
        db.refresh(teacher)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Teacher email already exists")
    return {"id": teacher.id, "name": teacher.name, "email": teacher.email, "department_id": teacher.department_id}


@app.put("/api/teachers/{teacher_id}")
def update_teacher(
    teacher_id: int,
    payload: TeacherUpdate,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_role("admin")),
):
    teacher = db.get(Teacher, teacher_id)
    if teacher is None:
        raise HTTPException(status_code=404, detail="Teacher not found")
    department = db.get(Department, payload.department_id)
    if department is None:
        raise HTTPException(status_code=404, detail="Department not found")
    name, email = payload.name.strip(), str(payload.email).strip().lower()
    duplicate = db.query(Teacher).filter(
        func.lower(Teacher.email) == email, Teacher.id != teacher_id
    ).first()
    if not name:
        raise HTTPException(status_code=400, detail="Teacher name is required")
    if duplicate:
        raise HTTPException(status_code=409, detail="Teacher email already exists")
    if teacher.department_id != department.id and (
        db.query(AttendanceSession.id).filter(AttendanceSession.teacher_id == teacher_id).first()
        or db.query(Subject.id).filter(Subject.teacher_id == teacher_id).first()
    ):
        raise HTTPException(status_code=409, detail="A teacher with session or subject assignments cannot be moved to another department")
    linked_user = db.query(User).filter(
        User.role == "teacher",
        ((User.teacher_id == teacher.id) | ((User.teacher_id.is_(None)) & (func.lower(User.email) == teacher.email.lower())))
    ).first()
    email_owner = db.query(User).filter(func.lower(User.email) == email).first()
    if email_owner and email_owner.role != "teacher":
        raise HTTPException(status_code=409, detail="New email belongs to a non-teacher account")
    if email_owner and email_owner.teacher_id not in {None, teacher.id}:
        raise HTTPException(status_code=409, detail="New email is linked to a different teacher profile")
    if email_owner and linked_user and email_owner.id != linked_user.id:
        raise HTTPException(status_code=409, detail="New email belongs to a different teacher account")
    if email_owner and linked_user is None:
        # Preserve the account's sign-in email while recording the explicit link.
        linked_user = email_owner
    if linked_user and linked_user.email.lower() != email:
        linked_user.email = email
    if linked_user:
        linked_user.teacher_id = teacher.id
    teacher.name, teacher.email, teacher.department_id = name, email, department.id
    db.commit()
    return {"id": teacher.id, "name": teacher.name, "email": teacher.email, "department_id": teacher.department_id}


@app.get("/api/attendance/sessions")
def get_attendance_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role == "student":
        raise HTTPException(status_code=403, detail="Staff access required")
    sessions = _visible_sessions(db, current_user)
    return [
        {
            "id": item.id,
            "subject_id": item.subject_id,
            "subject": item.subject.name,
            "teacher_id": item.teacher_id,
            "teacher": item.teacher.name,
            "session_date": item.session_date.isoformat(),
            "start_time": item.start_time.isoformat(),
            "end_time": item.end_time.isoformat() if item.end_time else None,
            "status": item.status,
            "check_ins": sum(record.status == "present" for record in item.attendance_records),
            "attendance_total": len(item.attendance_records),
        }
        for item in sessions
    ]


@app.get("/api/attendance/records")
def get_attendance_records(
    session_id: int | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    subject_id: int | None = None,
    student_id: int | None = None,
    status: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(AttendanceRecord).join(AttendanceSession)
    if current_user.role == "teacher":
        teacher = _teacher_for_user(db, current_user)
        query = query.filter(AttendanceSession.teacher_id == teacher.id)
    elif current_user.role == "student":
        student = _student_for_user(db, current_user)
        query = query.filter(AttendanceRecord.student_id == student.id)
    elif current_user.role != "admin":
        raise HTTPException(status_code=403, detail="You are not authorized to view attendance records")
    if session_id is not None:
        query = query.filter(AttendanceRecord.session_id == session_id)
    if subject_id is not None:
        query = query.filter(AttendanceSession.subject_id == subject_id)
    if student_id is not None:
        if current_user.role == "student" and student_id != _student_for_user(db, current_user).id:
            raise HTTPException(status_code=403, detail="You cannot view another student's records")
        query = query.filter(AttendanceRecord.student_id == student_id)
    if status is not None:
        normalized_status = status.strip().lower()
        if normalized_status not in {"present", "absent"}:
            raise HTTPException(status_code=400, detail="status must be present or absent")
        query = query.filter(AttendanceRecord.status == normalized_status)
    if start_date is not None:
        query = query.filter(AttendanceRecord.marked_at >= datetime.combine(start_date, datetime.min.time()))
    if end_date is not None:
        query = query.filter(AttendanceRecord.marked_at < datetime.combine(end_date + timedelta(days=1), datetime.min.time()))
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be on or before end_date")
    records = query.order_by(AttendanceRecord.marked_at.desc()).all()
    return [
        {
            "id": record.id,
            "session_id": record.session_id,
            "student_id": record.student_id,
            "student_name": record.student.name,
            "roll_number": record.student.roll_number,
            "subject": record.session.subject.name,
            "teacher": record.session.teacher.name,
            "session_date": record.session.session_date.isoformat(),
            "status": record.status,
            "marked_at": record.marked_at.replace(tzinfo=timezone.utc).isoformat(),
            "method": record.recognition_method,
            "recognition_distance": record.recognition_distance,
            "marked_by": record.marked_by.email if record.marked_by else None,
            "audit": [
                {"action": event.action, "previous_status": event.previous_status,
                 "new_status": event.new_status, "note": event.note,
                 "actor": event.actor.email, "changed_at": event.changed_at.isoformat()}
                for event in record.audit_events
            ],
        }
        for record in records
    ]


@app.get("/api/attendance/records.csv")
def export_attendance_records(
    session_id: int | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    subject_id: int | None = None,
    student_id: int | None = None,
    status: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = get_attendance_records(
        session_id=session_id, start_date=start_date, end_date=end_date,
        subject_id=subject_id, student_id=student_id, status=status,
        db=db, current_user=current_user,
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=[
        "id", "session_id", "session_date", "marked_at", "student_id", "student_name",
        "roll_number", "subject", "teacher", "status", "method", "recognition_distance", "marked_by",
    ])
    writer.writeheader()
    for row in rows:
        safe_row = {}
        for key in writer.fieldnames:
            value = row[key]
            text_value = "" if value is None else str(value)
            safe_row[key] = f"'{text_value}" if text_value.startswith(("=", "+", "-", "@", "\t", "\r")) else text_value
        writer.writerow(safe_row)
    return Response(
        content=stream.getvalue(), media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=smartattend-records.csv"},
    )


def _build_analytics(db: Session, current_user: User) -> dict:
    if current_user.role == "teacher":
        teacher = _teacher_for_user(db, current_user)
        students = db.query(Student).filter(Student.department_id == teacher.department_id).all()
        sessions = db.query(AttendanceSession).join(Subject).filter(
            AttendanceSession.teacher_id == teacher.id
        ).order_by(AttendanceSession.start_time.desc()).all()
    elif current_user.role == "student":
        student = _student_for_user(db, current_user)
        students = [student]
        sessions = db.query(AttendanceSession).join(Subject).filter(
            Subject.department_id == student.department_id
        ).order_by(AttendanceSession.start_time.desc()).all()
    elif current_user.role == "admin":
        students = db.query(Student).order_by(Student.name).all()
        sessions = db.query(AttendanceSession).join(Subject).order_by(
            AttendanceSession.start_time.desc()
        ).all()
    else:
        raise HTTPException(status_code=403, detail="You are not authorized to view analytics")

    session_ids = [session.id for session in sessions]
    record_query = db.query(AttendanceRecord).filter(
        AttendanceRecord.session_id.in_(session_ids) if session_ids else False
    )
    if current_user.role == "student":
        record_query = record_query.filter(AttendanceRecord.student_id == students[0].id)
    records = record_query.all()
    completed = [session for session in sessions if session.status == "completed"]
    completed_ids = {session.id for session in completed}
    department_totals: dict[int, int] = {}
    subject_totals: dict[int, int] = {}
    for session in completed:
        department_id = session.subject.department_id
        department_totals[department_id] = department_totals.get(department_id, 0) + 1
        subject_totals[session.subject_id] = subject_totals.get(session.subject_id, 0) + 1

    valid_records = [record for record in records if record.session_id in completed_ids]
    session_by_id = {session.id: session for session in completed}
    students_by_id = {student.id: student for student in students}
    attended_by_student: dict[int, int] = {}
    attended_by_subject: dict[int, int] = {}
    for record in valid_records:
        if record.student_id not in students_by_id:
            continue
        session = session_by_id.get(record.session_id)
        if session is None:
            continue
        if record.status == "present":
            attended_by_student[record.student_id] = attended_by_student.get(record.student_id, 0) + 1
            attended_by_subject[session.subject_id] = attended_by_subject.get(session.subject_id, 0) + 1

    student_rows = []
    denominator_total = 0
    numerator_total = 0
    for student in students:
        denominator = department_totals.get(student.department_id, 0)
        attended = attended_by_student.get(student.id, 0)
        percentage = round(attended * 100 / denominator, 1) if denominator else None
        denominator_total += denominator
        numerator_total += min(attended, denominator)
        student_rows.append({
            "student_id": student.id,
            "student_name": student.name,
            "roll_number": student.roll_number,
            "department_id": student.department_id,
            "attended_sessions": attended,
            "eligible_completed_sessions": denominator,
            "attendance_percentage": percentage,
            "at_risk": bool(denominator and percentage < ATTENDANCE_THRESHOLD_PERCENT),
        })

    subjects = {}
    for session in sessions:
        subjects[session.subject_id] = session.subject
    subject_rows = []
    students_per_department: dict[int, int] = {}
    for student in students:
        students_per_department[student.department_id] = students_per_department.get(student.department_id, 0) + 1
    for subject_id, subject in subjects.items():
        completed_subject_sessions = subject_totals.get(subject_id, 0)
        if current_user.role == "student":
            denominator = completed_subject_sessions
        else:
            denominator = completed_subject_sessions * students_per_department.get(subject.department_id, 0)
        attended = attended_by_subject.get(subject_id, 0)
        subject_rows.append({
            "subject_id": subject_id,
            "subject": subject.name,
            "code": subject.code,
            "completed_sessions": completed_subject_sessions,
            "eligible_attendance_opportunities": denominator,
            "present_records": attended,
            "attendance_percentage": min(100.0, round(attended * 100 / denominator, 1)) if denominator else None,
        })
    subject_rows.sort(key=lambda item: item["subject"].lower())

    today = utc_now_naive().date()
    daily = []
    for offset in range(13, -1, -1):
        day = today - timedelta(days=offset)
        daily.append({
            "date": day.isoformat(),
            "count": sum(record.status == "present" and record.marked_at.date() == day for record in records),
        })

    current_enrollments = sum(embedding_is_current(student.face_embedding) for student in students)
    session_rows = []
    for session in sessions[:8]:
        session_records = [record for record in records if record.session_id == session.id]
        session_rows.append({
            "id": session.id,
            "subject": session.subject.name,
            "teacher": session.teacher.name,
            "date": session.session_date.isoformat(),
            "status": session.status,
            "check_ins": sum(record.status == "present" for record in session_records),
            "record_count": len(session_records),
        })

    attendance_percentage = round(numerator_total * 100 / denominator_total, 1) if denominator_total else None
    return {
        "role": current_user.role,
        "student_count": len(students),
        "enrolled_student_count": current_enrollments,
        "enrollment_coverage_percent": round(current_enrollments * 100 / len(students), 1) if students else None,
        "session_count": sum(session.status != "cancelled" for session in sessions),
        "completed_session_count": len(completed),
        "active_sessions": sum(session.status == "active" for session in sessions),
        "check_ins": sum(record.status == "present" for record in records),
        "check_ins_this_week": sum(record.status == "present" and record.marked_at.date() >= today - timedelta(days=6) for record in records),
        "attendance_percentage": attendance_percentage,
        "attendance_threshold_percent": ATTENDANCE_THRESHOLD_PERCENT,
        "attendance_denominator": denominator_total,
        "defaulter_count": sum(item["at_risk"] for item in student_rows),
        "defaulters": sorted(
            [item for item in student_rows if item["at_risk"]],
            key=lambda item: (item["attendance_percentage"], item["student_name"].lower()),
        ),
        "student_attendance": student_rows,
        "subject_attendance": subject_rows,
        "recent_sessions": session_rows,
        "daily_check_ins": daily,
        "attendance_policy": (
            "Denominator is the student's department's completed sessions. Sessions in progress are excluded; "
            "a closed session is finalized. A missing check-in contributes no attended session, but no absent "
            "attendance record is created unless staff explicitly enters one. Because there is no student-subject "
            "registration table, every student in a department is treated as eligible for each session in that department."
        ),
    }


@app.get("/api/dashboard")
def get_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _build_analytics(db, current_user)


@app.get("/api/analytics")
def get_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _build_analytics(db, current_user)


@app.post("/api/attendance/sessions/{session_id}/manual")
def manually_record_attendance(
    session_id: int,
    payload: ManualAttendanceRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role not in {"admin", "teacher"}:
        raise HTTPException(status_code=403, detail="Staff access required")
    session = db.get(AttendanceSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if current_user.role == "teacher":
        teacher = _teacher_for_user(db, current_user)
        if session.teacher_id != teacher.id:
            raise HTTPException(status_code=403, detail="This is not your session")
    student = db.get(Student, payload.student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    if student.department_id != session.subject.department_id:
        raise HTTPException(status_code=400, detail="Student is not eligible for this session")
    try:
        record, action = record_manual_attendance(
            db, session, student, payload.status, current_user.id, payload.note
        )
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Attendance could not be updated; reload the session")
    return {
        "saved": True,
        "record_id": record.id,
        "student_id": student.id,
        "student_name": student.name,
        "status": record.status,
        "method": record.recognition_method,
        "action": action,
        "message": "Manual attendance saved and recorded in the audit history.",
    }


# =========================================================
# USER REGISTRATION API
# =========================================================

@app.post("/api/register", response_model=UserResponse)
def register_user(
    user_data: RegisterRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    if len(user_data.password) < 12 or len(user_data.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password must be 12 to 72 UTF-8 bytes")
    # Account creation is restricted to an authenticated administrator.
    email = str(user_data.email).strip().lower()
    existing_user = db.query(User).filter(func.lower(User.email) == email).first()

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered",
        )

    linked_teacher = None
    if user_data.role == "teacher":
        linked_teacher = db.query(Teacher).filter(func.lower(Teacher.email) == email).first()
        if linked_teacher is None:
            raise HTTPException(status_code=400, detail="Create the matching teacher record before creating an account")
    linked_student = None
    if user_data.role == "student":
        linked_student = db.query(Student).filter(func.lower(Student.email) == email).first()
        if linked_student is None:
            raise HTTPException(status_code=400, detail="Create the matching student record before creating an account")
        if db.query(User.id).filter(
            User.role == "student",
            ((User.student_id == linked_student.id) | (func.lower(User.email) == email)),
        ).first():
            raise HTTPException(status_code=409, detail="This student already has a login account")

    new_user = User(
        email=email,
        password_hash=hash_password(user_data.password),
        role=user_data.role,
        student_id=linked_student.id if linked_student else None,
        teacher_id=linked_teacher.id if linked_teacher else None,
        username=_unique_student_username(db, linked_student.roll_number) if linked_student else None,
        must_change_password=linked_student is not None,
    )

    db.add(new_user)

    try:
        db.commit()
        db.refresh(new_user)
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not create the user account",
        )

    return new_user


# =========================================================
# LOGIN API
# =========================================================

@app.post("/api/login")
def login_user(
    user_data: LoginRequest,
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter((func.lower(User.email) == str(user_data.email).strip().lower()) |
                (func.lower(User.username) == str(user_data.email).strip().lower()))
        .first()
    )

    if not user or not verify_password(
        user_data.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
        )

    access_token = create_access_token(
        {
            "sub": str(user.id),
            "role": user.role,
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "role": user.role,
            "must_change_password": user.must_change_password,
        },
    }


# =========================================================
# TEACHER AUTHORIZATION TEST
# =========================================================

@app.get("/api/teacher-test")
def teacher_test(
    current_user: User = Depends(require_role("teacher")),
):
    return {
        "message": "Teacher authorization successful",
        "user_id": current_user.id,
        "email": current_user.email,
        "role": current_user.role,
    }


@app.post("/api/students/enroll-face")
async def enroll_student_face(
    name: str | None = Form(None),
    roll_number: str | None = Form(None),
    email: str | None = Form(None),
    department_id: int | None = Form(None),
    student_id: int | None = Form(None),
    images: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Only authorized staff can enroll students.
    if current_user.role not in {"admin", "teacher"}:
        raise HTTPException(status_code=403, detail="Only staff can enroll students")

    if len(images) != 30:
        raise HTTPException(status_code=400, detail="Submit exactly 30 face samples")

    existing_student = None
    existing_embedding = None
    if student_id is not None:
        existing_student = db.query(Student).filter(Student.id == student_id).first()
        if existing_student is None:
            raise HTTPException(status_code=404, detail="Student not found")
        existing_embedding = existing_student.face_embedding
        if embedding_is_current(existing_embedding):
            raise HTTPException(status_code=409, detail="This student already has a valid face enrollment")
        if department_id is not None and department_id != existing_student.department_id:
            raise HTTPException(status_code=400, detail="Department does not match the selected student")
        department_id = existing_student.department_id
    else:
        name = (name or "").strip()
        roll_number = (roll_number or "").strip()
        email = (email or "").strip().lower()
        if not name or not roll_number or not email:
            raise HTTPException(status_code=400, detail="Name, roll number and email are required for a new student")
        try:
            from email_validator import validate_email
            email = validate_email(email, check_deliverability=False).normalized
        except Exception:
            raise HTTPException(status_code=400, detail="A valid student email address is required")
        if db.query(Student).filter(
            (Student.roll_number == roll_number) | (Student.email == email)
        ).first():
            raise HTTPException(status_code=409, detail="Roll number or email already exists; select the existing student instead")
        if current_user.role == "admin" and db.query(User.id).filter(func.lower(User.email) == email).first():
            raise HTTPException(status_code=409, detail="Student email is already used by another account")

    if department_id is None:
        raise HTTPException(status_code=400, detail="Department is required")

    if not db.query(Department).filter(Department.id == department_id).first():
        raise HTTPException(status_code=400, detail="Department does not exist")

    if current_user.role == "teacher":
        teacher = _teacher_for_user(db, current_user)
        if teacher.department_id != department_id:
            raise HTTPException(status_code=403, detail="You cannot enroll students in this department")

    try:
        # Load the actual face-quality pipeline.
        recognizer = FaceRecognition()

        vectors = []
        rejected_samples = []

        for index, uploaded in enumerate(images, start=1):
            if uploaded.content_type not in {"image/jpeg", "image/png", "image/webp"}:
                rejected_samples.append({"index": index, "reason": "Unsupported image type"})
                continue
            raw = await uploaded.read()

            if not raw:
                rejected_samples.append({"index": index, "reason": "Image is empty"})
                continue

            # Limit each uploaded image to 8 MB.
            if len(raw) > 8 * 1024 * 1024:
                rejected_samples.append({"index": index, "reason": "Image exceeds 8 MB"})
                continue

            frame = cv2.imdecode(
                np.frombuffer(raw, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is None or frame.shape[0] * frame.shape[1] > 16_000_000:
                rejected_samples.append({"index": index, "reason": "Invalid image or image exceeds 16 megapixels"})
                continue

            quality = recognizer.preprocess_frame(frame)

            if not quality.get("success") or quality.get("face") is None:
                rejected_samples.append({"index": index, "reason": quality.get("message", "Face quality check failed")})
                continue

            try:
                vector = recognizer.embed_aligned_face(quality["face"])
            except ValueError as exc:
                rejected_samples.append({"index": index, "reason": str(exc)})
                continue

            vectors.append(vector)

        # A few poor captures should not force the user to retake every frame,
        # but never save an enrollment with too few usable samples.
        dimensions = {vector.shape[0] for vector in vectors}
        if len(vectors) < 10 or len(dimensions) != 1:
            rejection_summary = "; ".join(
                f"sample {item['index']}: {item['reason']}"
                for item in rejected_samples[:5]
            )
            raise HTTPException(
                status_code=422,
                detail=f"Only {len(vectors)} usable face samples; at least 10 are required. {rejection_summary}"
            )

        # Store all 30 real embeddings in one binary database record.
        embedding_matrix = np.stack(vectors).astype(EMBEDDING_DTYPE, copy=False)
        embedding_bytes = embedding_matrix.tobytes()

        student = existing_student
        initial_credentials = None
        if student is None:
            student = Student(
                name=name,
                roll_number=roll_number,
                email=email,
                department_id=department_id,
            )
            db.add(student)
            db.flush()  # Get the database-generated student ID.
            if current_user.role == "admin":
                username = _unique_student_username(db, roll_number)
                temporary_password = secrets.token_urlsafe(18)
                db.add(User(email=email, username=username, student_id=student.id,
                            password_hash=hash_password(temporary_password), role="student",
                            must_change_password=True))
                initial_credentials = {"username": username, "email": email,
                                       "temporary_password": temporary_password}

        if existing_embedding is None:
            existing_embedding = FaceEmbedding(student_id=student.id)
            db.add(existing_embedding)
        existing_embedding.embedding = embedding_bytes
        existing_embedding.model_name = MODEL_NAME
        existing_embedding.model_version = MODEL_VERSION
        existing_embedding.vector_dimension = VECTOR_DIMENSION
        existing_embedding.preprocessing = PREPROCESSING_METADATA

        db.commit()
        db.refresh(student)

        return {
            "success": True,
            "student_id": student.id,
            "name": student.name,
            "embeddings_processed": len(vectors),
            "samples_rejected": len(rejected_samples),
            "initial_credentials": initial_credentials,
            "rejected_sample_details": rejected_samples[:10],
            "embedding_dimensions": int(embedding_matrix.shape[1]),
            "message": f"Student saved with {len(vectors)} valid face embeddings; {len(rejected_samples)} samples rejected.",
        }

    except HTTPException:
        db.rollback()
        raise
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Student or face enrollment already exists")
    except ImportError as exc:
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail=f"Required AI dependency is unavailable: {exc}"
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Enrollment failed during processing. No student or embedding was saved."
        )
    finally:
        for uploaded in images:
            await uploaded.close()


@app.post("/api/attendance/sessions/{session_id}/recognize")
async def recognize_attendance(
    session_id: int,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role not in {"admin", "teacher"}:
        raise HTTPException(status_code=403, detail="Staff access required")

    session = db.query(AttendanceSession).filter(
        AttendanceSession.id == session_id,
        AttendanceSession.status == "active",
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Active attendance session not found")

    if current_user.role == "teacher":
        teacher = _teacher_for_user(db, current_user)
        if session.teacher_id != teacher.id:
            raise HTTPException(status_code=403, detail="This is not your session")

    try:
        if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status_code=415, detail="Frame must be JPEG, PNG or WebP")
        raw = await image.read()
        if not raw or len(raw) > 8 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Invalid image or image exceeds 8 MB")

        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[0] * frame.shape[1] > 16_000_000:
            raise HTTPException(status_code=400, detail="Invalid image data or image exceeds the 16-megapixel limit")

        recognizer = FaceRecognition()
        processed = recognizer.preprocess_frame(frame)
        if not processed.get("success") or processed.get("face") is None:
            return {"recognized": False, "reason": processed.get("message", "Face not detected")}

        probe = recognizer.embed_aligned_face(processed["face"])
        enrolled = db.query(FaceEmbedding, Student).join(
            Student, Student.id == FaceEmbedding.student_id
        ).filter(
            Student.department_id == session.subject.department_id,
            FaceEmbedding.model_name == MODEL_NAME,
            FaceEmbedding.model_version == MODEL_VERSION,
            FaceEmbedding.vector_dimension == VECTOR_DIMENSION,
            FaceEmbedding.preprocessing == PREPROCESSING_METADATA,
        ).all()
        best_student, best_distance, match_error = choose_face_match(
            probe,
            [(student, stored.embedding) for stored, student in enrolled],
        )
        if best_student is None:
            return {
                "recognized": False,
                "reason": match_error,
                "distance_metric": "cosine",
                "maximum_distance": MAX_COSINE_DISTANCE,
                "minimum_match_margin": MIN_MATCH_MARGIN,
            }
        if best_student.department_id != session.subject.department_id:
            return {"recognized": False, "reason": "Student is not eligible for this session"}

        try:
            _record, already_marked = record_check_in(
                db, session, best_student, best_distance, current_user.id
            )
        except StudentNotEligibleError as exc:
            return {"recognized": False, "reason": str(exc)}
        except SessionNotActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except AttendanceConflictError as exc:
            db.rollback()
            return {"recognized": False, "reason": str(exc)}

        if already_marked:
            return {
                "recognized": True,
                "already_marked": True,
                "student_id": best_student.id,
                "student_name": best_student.name,
            }

        return {
            "recognized": True,
            "already_marked": False,
            "student_id": best_student.id,
            "student_name": best_student.name,
            "distance": round(best_distance, 4),
            "method": "face",
            "distance_metric": "cosine",
            "maximum_distance": MAX_COSINE_DISTANCE,
            "minimum_match_margin": MIN_MATCH_MARGIN,
        }

    except HTTPException:
        db.rollback()
        raise
    except ImportError:
        db.rollback()
        raise HTTPException(status_code=503, detail="DeepFace is not installed in this environment")
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Recognition processing failed. No attendance was recorded.")
    finally:
        await image.close()

@app.post("/api/attendance/sessions")
def create_attendance_session(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role not in {"admin", "teacher"}:
        raise HTTPException(status_code=403, detail="Staff access required")

    subject_id = payload.get("subject_id")

    if not isinstance(subject_id, int) or isinstance(subject_id, bool):
        raise HTTPException(status_code=400, detail="A valid subject_id is required")

    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if subject is None:
        raise HTTPException(status_code=404, detail="Subject not found")

    # A teacher must belong to the subject's department.
    if current_user.role == "teacher":
        teacher = _teacher_for_user(db, current_user)

        if teacher.department_id != subject.department_id:
            raise HTTPException(
                status_code=403,
                detail="You cannot start a session for this department",
            )
        has_explicit_subjects = db.query(Subject.id).filter(
            Subject.department_id == teacher.department_id,
            Subject.teacher_id == teacher.id,
        ).first()
        if subject.teacher_id != teacher.id and (has_explicit_subjects or subject.teacher_id is not None):
            raise HTTPException(status_code=403, detail="This subject is not assigned to your class")

        teacher_id = teacher.id
    else:
        teacher_id = payload.get("teacher_id")

        if not isinstance(teacher_id, int) or isinstance(teacher_id, bool):
            raise HTTPException(
                status_code=400,
                detail="Admin must provide a valid teacher_id",
            )

        teacher = db.query(Teacher).filter(
            Teacher.id == teacher_id
        ).first()

        if teacher is None:
            raise HTTPException(status_code=404, detail="Teacher not found")

        if teacher.department_id != subject.department_id:
            raise HTTPException(status_code=400, detail="Teacher and subject must belong to the same department")
        if subject.teacher_id is not None and subject.teacher_id != teacher.id:
            raise HTTPException(status_code=403, detail="This subject is assigned to a different teacher")

    active_session = db.query(AttendanceSession).filter(
        AttendanceSession.teacher_id == teacher_id,
        AttendanceSession.subject_id == subject.id,
        AttendanceSession.status == "active",
    ).first()
    if active_session:
        raise HTTPException(
            status_code=409,
            detail=f"Session {active_session.id} is already active for this class. Finish or cancel it from session history first.",
        )

    session_start = utc_now_naive()
    session = AttendanceSession(
        teacher_id=teacher_id,
        subject_id=subject.id,
        session_date=session_start.date(),
        start_time=session_start,
        status="active",
    )

    try:
        db.add(session)
        db.commit()
        db.refresh(session)

        return {
            "success": True,
            "session_id": session.id,
            "subject_id": session.subject_id,
            "teacher_id": session.teacher_id,
            "status": session.status,
            "message": "Attendance session created",
        }
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not save the attendance session",
        )


@app.post("/api/attendance/sessions/{session_id}/end")
def end_attendance_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role not in {"admin", "teacher"}:
        raise HTTPException(status_code=403, detail="Staff access required")

    session = db.query(AttendanceSession).filter(
        AttendanceSession.id == session_id
    ).first()

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != "active":
        raise HTTPException(
            status_code=409,
            detail="This session is already closed",
        )

    if current_user.role == "teacher":
        teacher = _teacher_for_user(db, current_user)
        if teacher.id != session.teacher_id:
            raise HTTPException(
                status_code=403,
                detail="You cannot close this session",
            )

    try:
        ended_at = utc_now_naive()
        updated = db.execute(
            update(AttendanceSession)
            .where(AttendanceSession.id == session_id, AttendanceSession.status == "active")
            .values(status="completed", end_time=ended_at)
        )
        if updated.rowcount != 1:
            db.rollback()
            raise HTTPException(status_code=409, detail="The session was already closed or cancelled")
        db.commit()
        db.refresh(session)
        return {
            "success": True,
            "session_id": session.id,
            "status": session.status,
            "end_time": session.end_time.isoformat(),
        }
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not close the session",
        )


@app.post("/api/attendance/sessions/{session_id}/cancel")
def cancel_empty_attendance_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role not in {"admin", "teacher"}:
        raise HTTPException(status_code=403, detail="Staff access required")
    session = db.get(AttendanceSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if current_user.role == "teacher":
        teacher = _teacher_for_user(db, current_user)
        if session.teacher_id != teacher.id:
            raise HTTPException(status_code=403, detail="You cannot cancel this session")
    ended_at = utc_now_naive()
    try:
        # Serialize with face check-ins: a cancelled session must remain empty.
        result = db.execute(text(
            "UPDATE attendance_sessions SET status='cancelled', end_time=:ended_at "
            "WHERE id=:session_id AND status='active' "
            "AND NOT EXISTS (SELECT 1 FROM attendance_records WHERE session_id=:session_id)"
        ), {"ended_at": ended_at, "session_id": session_id})
        if result.rowcount != 1:
            db.rollback()
            raise HTTPException(
                status_code=409,
                detail="Only an active session with no attendance records can be cancelled",
            )
        db.commit()
        return {"success": True, "session_id": session_id, "status": "cancelled",
                "end_time": ended_at.isoformat(),
                "message": "Empty session cancelled. It is excluded from attendance calculations."}
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not cancel the empty session")
