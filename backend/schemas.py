from enum import Enum

from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    ADMIN = "admin"
    TEACHER = "teacher"
    STUDENT = "student"


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    role: UserRole


class FirstAdminRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=72)


class LoginRequest(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=12, max_length=72)


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    role: UserRole
    username: str | None = None
    must_change_password: bool = False

    class Config:
        from_attributes = True


class DepartmentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class DepartmentUpdate(DepartmentCreate):
    pass


class TeacherCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    department_id: int = Field(gt=0)


class TeacherUpdate(TeacherCreate):
    pass


class SubjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    code: str = Field(min_length=1, max_length=32)
    department_id: int = Field(gt=0)


class SubjectUpdate(SubjectCreate):
    pass


class SingleClassSetupRequest(BaseModel):
    teacher_id: int = Field(gt=0)
    teacher_user_id: int = Field(gt=0)
    subject_id: int = Field(gt=0)


class StudentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    roll_number: str = Field(min_length=1, max_length=64)
    email: EmailStr
    department_id: int = Field(gt=0)


class StudentUpdate(StudentCreate):
    pass


class ManualAttendanceRequest(BaseModel):
    student_id: int = Field(gt=0)
    status: str = Field(pattern="^(present|absent)$")
    note: str = Field(default="", max_length=500)
