"""Create the first SmartAttend administrator without seeding other data."""

from getpass import getpass
from email_validator import validate_email, EmailNotValidError
from sqlalchemy import text

from backend.auth import hash_password
from backend.database import SessionLocal
from backend.models import User


def main() -> None:
    email = input("Administrator email: ").strip().lower()
    try:
        email = validate_email(email, check_deliverability=False).normalized
    except EmailNotValidError as exc:
        raise SystemExit(f"Invalid email address: {exc}")
    password = getpass("Administrator password (minimum 12 characters): ")
    confirmation = getpass("Confirm password: ")

    if len(password) < 12 or len(password.encode("utf-8")) > 72:
        raise SystemExit("Password must be 12 to 72 UTF-8 bytes.")
    if password != confirmation:
        raise SystemExit("Passwords do not match.")

    db = SessionLocal()
    try:
        db.execute(text("BEGIN IMMEDIATE"))
        if db.query(User).first():
            raise SystemExit("User accounts already exist; bootstrap is disabled.")
        db.add(User(email=email, password_hash=hash_password(password), role="admin"))
        db.commit()
        print(f"Administrator account created for {email}.")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
