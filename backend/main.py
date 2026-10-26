from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from backend.auth import hash_password, verify_password
from backend.database import get_db
from backend.dependencies import get_current_user, require_role
from backend.models import Student, User
from backend.schemas import (
    LoginRequest,
    RegisterRequest,
    UserResponse,
)
from backend.token import create_access_token


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

app = FastAPI(
    title="SmartAttend",
    description="Privacy-first attendance management system",
    version="1.0.0"
)


# =========================================================
# STATIC FILES
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# =========================================================
# JINJA2 TEMPLATES
# =========================================================

templates = Jinja2Templates(
    directory=TEMPLATES_DIR
)


# =========================================================
# PUBLIC HTML ROUTES
# =========================================================

@app.get("/")
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "active_page": "home"
        }
    )


@app.get("/login")
def login(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={}
    )


# =========================================================
# APPLICATION HTML ROUTES
# =========================================================

@app.get("/teacher-dashboard")
def teacher_dashboard(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="dashboards/teacher-dashboard.html",
        context={
            "active_page": "dashboard"
        }
    )


@app.get("/take-attendance")
def take_attendance(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="attendance/take-attendance.html",
        context={
            "active_page": "take-attendance"
        }
    )


@app.get("/attendance-records")
def attendance_records(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="attendance/attendance-records.html",
        context={
            "active_page": "records"
        }
    )


@app.get("/students")
def students(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="students/students.html",
        context={
            "active_page": "students"
        }
    )


@app.get("/analytics")
def analytics(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="analytics/analytics.html",
        context={
            "active_page": "analytics"
        }
    )


# =========================================================
# STUDENTS API
# =========================================================

@app.get("/api/students")
def get_students(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    students = db.query(Student).all()

    return students


# =========================================================
# REGISTER API
# =========================================================

@app.post(
    "/api/register",
    response_model=UserResponse
)
def register_user(
    user_data: RegisterRequest,
    db: Session = Depends(get_db)
):

    # Check whether email already exists
    existing_user = db.query(User).filter(
        User.email == user_data.email
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    # Create new user
    new_user = User(
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        role=user_data.role
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


# =========================================================
# LOGIN API
# =========================================================

@app.post("/api/login")
def login_user(
    user_data: LoginRequest,
    db: Session = Depends(get_db)
):

    # Find user by email
    user = db.query(User).filter(
        User.email == user_data.email
    ).first()

    # User doesn't exist
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    # Password doesn't match
    if not verify_password(
        user_data.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    # Create JWT
    access_token = create_access_token({
        "sub": str(user.id),
        "role": user.role
    })

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "role": user.role
        }
    }


# =========================================================
# TEACHER AUTHORIZATION TEST
# =========================================================

@app.get("/api/teacher-test")
def teacher_test(
    current_user: User = Depends(require_role("teacher"))
):

    return {
        "message": "You are authorized as a teacher",
        "user_id": current_user.id,
        "email": current_user.email,
        "role": current_user.role
    }