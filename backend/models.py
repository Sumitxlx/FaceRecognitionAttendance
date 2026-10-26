from datetime import datetime, date

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# =========================================================
# USER
# =========================================================

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    email: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True
    )

    password_hash: Mapped[str] = mapped_column(
        String,
        nullable=False
    )

    role: Mapped[str] = mapped_column(
        String,
        nullable=False
    )


# =========================================================
# DEPARTMENT
# =========================================================

class Department(Base):
    __tablename__ = "departments"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True
    )

    teachers = relationship(
        "Teacher",
        back_populates="department"
    )

    students = relationship(
        "Student",
        back_populates="department"
    )

    subjects = relationship(
        "Subject",
        back_populates="department"
    )


# =========================================================
# TEACHER
# =========================================================

class Teacher(Base):
    __tablename__ = "teachers"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    name: Mapped[str] = mapped_column(
        String,
        nullable=False
    )

    email: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True
    )

    department_id: Mapped[int] = mapped_column(
        ForeignKey("departments.id"),
        nullable=False
    )

    department = relationship(
        "Department",
        back_populates="teachers"
    )

    sessions = relationship(
        "AttendanceSession",
        back_populates="teacher"
    )


# =========================================================
# STUDENT
# =========================================================

class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    name: Mapped[str] = mapped_column(
        String,
        nullable=False
    )

    roll_number: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True
    )

    email: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True
    )

    department_id: Mapped[int] = mapped_column(
        ForeignKey("departments.id"),
        nullable=False
    )

    department = relationship(
        "Department",
        back_populates="students"
    )

    attendance_records = relationship(
        "AttendanceRecord",
        back_populates="student"
    )

    face_embedding = relationship(
        "FaceEmbedding",
        back_populates="student",
        uselist=False
    )

    weekly_snapshots = relationship(
        "WeeklyAttendanceSnapshot",
        back_populates="student"
    )


# =========================================================
# SUBJECT
# =========================================================

class Subject(Base):
    __tablename__ = "subjects"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    name: Mapped[str] = mapped_column(
        String,
        nullable=False
    )

    code: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True
    )

    department_id: Mapped[int] = mapped_column(
        ForeignKey("departments.id"),
        nullable=False
    )

    department = relationship(
        "Department",
        back_populates="subjects"
    )

    sessions = relationship(
        "AttendanceSession",
        back_populates="subject"
    )


# =========================================================
# ATTENDANCE SESSION
# =========================================================

class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    teacher_id: Mapped[int] = mapped_column(
        ForeignKey("teachers.id"),
        nullable=False
    )

    subject_id: Mapped[int] = mapped_column(
        ForeignKey("subjects.id"),
        nullable=False
    )

    session_date: Mapped[date] = mapped_column(
        Date,
        nullable=False
    )

    start_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False
    )

    end_time: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )

    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="active"
    )

    teacher = relationship(
        "Teacher",
        back_populates="sessions"
    )

    subject = relationship(
        "Subject",
        back_populates="sessions"
    )

    attendance_records = relationship(
        "AttendanceRecord",
        back_populates="session"
    )


# =========================================================
# ATTENDANCE RECORD
# =========================================================

class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    session_id: Mapped[int] = mapped_column(
        ForeignKey("attendance_sessions.id"),
        nullable=False
    )

    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id"),
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String,
        nullable=False
    )

    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    marked_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    session = relationship(
        "AttendanceSession",
        back_populates="attendance_records"
    )

    student = relationship(
        "Student",
        back_populates="attendance_records"
    )

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "student_id",
            name="unique_student_per_session"
        ),
    )


# =========================================================
# FACE EMBEDDING
# =========================================================

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id"),
        nullable=False,
        unique=True
    )

    # Encryption will be implemented in Stage 6.
    embedding: Mapped[bytes] = mapped_column(
        LargeBinary,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    student = relationship(
        "Student",
        back_populates="face_embedding"
    )


# =========================================================
# WEEKLY ATTENDANCE SNAPSHOT
# =========================================================

class WeeklyAttendanceSnapshot(Base):
    __tablename__ = "weekly_attendance_snapshots"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id"),
        nullable=False
    )

    week_start: Mapped[date] = mapped_column(
        Date,
        nullable=False
    )

    attendance_percentage: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    student = relationship(
        "Student",
        back_populates="weekly_snapshots"
    )