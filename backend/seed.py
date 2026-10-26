from datetime import datetime, date

from backend.database import engine, SessionLocal
from backend.models import (
    Base,
    Department,
    Teacher,
    Student,
    Subject,
)


# Create all database tables
Base.metadata.create_all(bind=engine)


def seed_database():
    db = SessionLocal()

    try:
        # Check whether data already exists
        if db.query(Department).first():
            print("Database already contains data.")
            return

        # Department
        department = Department(
            name="Computer Science"
        )

        db.add(department)
        db.flush()

        # Teacher
        teacher = Teacher(
            name="Demo Teacher",
            email="teacher@smartattend.com",
            department_id=department.id
        )

        db.add(teacher)

        # Subject
        subject = Subject(
            name="Database Systems",
            code="CS301",
            department_id=department.id
        )

        db.add(subject)

        # Students
        students = [
            Student(
                name="Himanshu",
                roll_number="CS001",
                email="himanshu@smartattend.com",
                department_id=department.id
            ),
            Student(
                name="Priya",
                roll_number="CS002",
                email="priya@smartattend.com",
                department_id=department.id
            ),
            Student(
                name="Sumit",
                roll_number="CS003",
                email="sumit@smartattend.com",
                department_id=department.id
            ),
        ]

        db.add_all(students)

        # Save everything
        db.commit()

        print("Database seeded successfully.")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    seed_database()