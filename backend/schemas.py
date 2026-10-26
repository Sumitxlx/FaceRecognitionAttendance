from enum import Enum

from pydantic import BaseModel, EmailStr


class UserRole(str, Enum):
    ADMIN = "admin"
    TEACHER = "teacher"
    STUDENT = "student"


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    role: UserRole


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    role: UserRole

    class Config:
        from_attributes = True