"""Transactional attendance check-in operations shared by API handlers."""

from sqlalchemy.exc import IntegrityError
from sqlalchemy import update
from sqlalchemy.orm import Session

from backend.models import AttendanceAudit, AttendanceRecord, AttendanceSession, Student
from backend.time_utils import utc_now_naive


class SessionNotActiveError(ValueError):
    pass


class StudentNotEligibleError(ValueError):
    pass


class AttendanceConflictError(ValueError):
    pass


def record_check_in(
    db: Session,
    session: AttendanceSession,
    student: Student,
    distance: float,
    actor_user_id: int | None = None,
) -> tuple[AttendanceRecord, bool]:
    """Persist one confirmed check-in; return (record, already_marked)."""
    if session.status != "active":
        raise SessionNotActiveError("Attendance session is not active")
    if student.department_id != session.subject.department_id:
        raise StudentNotEligibleError("Student is not eligible for this session")

    # A conditional no-op UPDATE acquires SQLite's write lock and verifies that
    # a concurrent session close has not already committed.
    active = db.execute(
        update(AttendanceSession)
        .where(AttendanceSession.id == session.id, AttendanceSession.status == "active")
        .values(status=AttendanceSession.status)
    )
    if active.rowcount != 1:
        db.rollback()
        raise SessionNotActiveError("Attendance session is not active")

    existing = db.query(AttendanceRecord).filter_by(
        session_id=session.id,
        student_id=student.id,
    ).first()
    if existing:
        if existing.status != "present":
            raise AttendanceConflictError(
                "This student has an explicit non-present record; staff correction is required"
            )
        return existing, True

    record = AttendanceRecord(
        session_id=session.id,
        student_id=student.id,
        status="present",
        confidence=None,
        marked_at=utc_now_naive(),
        recognition_method="face",
        recognition_distance=distance,
        marked_by_user_id=actor_user_id,
    )
    db.add(record)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        # The unique (session_id, student_id) index resolves simultaneous frames.
        existing = db.query(AttendanceRecord).filter_by(
            session_id=session.id,
            student_id=student.id,
        ).first()
        if existing:
            if existing.status != "present":
                raise AttendanceConflictError(
                    "This student has an explicit non-present record; staff correction is required"
                )
            return existing, True
        raise
    db.refresh(record)
    return record, False


def record_manual_attendance(
    db: Session,
    session: AttendanceSession,
    student: Student,
    status: str,
    actor_user_id: int,
    note: str = "",
) -> tuple[AttendanceRecord, str]:
    """Create or audit a staff-entered attendance value (never an AI result)."""
    if status not in {"present", "absent"}:
        raise ValueError("Manual attendance status must be present or absent")
    if student.department_id != session.subject.department_id:
        raise StudentNotEligibleError("Student is not eligible for this session")

    record = db.query(AttendanceRecord).filter_by(
        session_id=session.id, student_id=student.id
    ).first()
    if record is None:
        record = AttendanceRecord(
            session_id=session.id,
            student_id=student.id,
            status=status,
            confidence=None,
            marked_at=utc_now_naive(),
            recognition_method="manual",
            recognition_distance=None,
            marked_by_user_id=actor_user_id,
        )
        db.add(record)
        db.flush()
        action, previous = "manual-entry", None
    elif record.status == status and record.recognition_method.startswith("manual"):
        return record, "unchanged"
    elif record.status == status:
        return record, "unchanged"
    else:
        previous = record.status
        action = "manual-correction"
        record.status = status
        record.recognition_method = "manual-correction"
        record.recognition_distance = None
        record.confidence = None
        record.marked_by_user_id = actor_user_id
        record.marked_at = utc_now_naive()

    db.add(AttendanceAudit(
        attendance_record_id=record.id,
        actor_user_id=actor_user_id,
        action=action,
        previous_status=previous,
        new_status=status,
        note=note.strip(),
    ))
    try:
        db.commit()
        db.refresh(record)
    except IntegrityError:
        db.rollback()
        record = db.query(AttendanceRecord).filter_by(
            session_id=session.id, student_id=student.id
        ).first()
        if record and record.status == status:
            return record, "unchanged"
        raise
    return record, action
