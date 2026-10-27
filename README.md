# SmartAttend

SmartAttend is a FastAPI attendance application with a browser GUI, SQLite storage, webcam capture, OpenCV face quality checks, and DeepFace ArcFace embeddings.

## Setup and launch

Use Python 3.11. From the project root on macOS:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r backend/requirements.txt
python -m backend.init_db
export SMARTATTEND_SECRET_KEY="$(openssl rand -hex 32)"
uvicorn backend.main:app --reload
```

Open <http://127.0.0.1:8000/login>. When no admin exists, login links to the first-admin setup page. That page is limited to localhost and becomes unavailable once an admin has been created. The first admin can also be created from a terminal with `python -m backend.create_admin`; that command only works while the users table is empty. Both flows hash the password and do not create demo records.

Keep `SMARTATTEND_SECRET_KEY` private and reuse the same value on future launches. If it changes, existing tokens become invalid. If the project `venv` is already installed, skip the first three commands.

The default database is `smartattend.db` at the project root. `python -m backend.init_db` creates missing tables and applies additive SQLite updates; it does not drop or recreate tables or rows. The application applies the same safe updates at startup. To use an isolated database, set `SMARTATTEND_DATABASE_URL`, for example `sqlite:////private/tmp/smartattend-test.db`, before initialization and launch.

Before an additive migration changes an existing SQLite file, SmartAttend writes a timestamped `.pre-migration-*.bak` copy beside it and verifies the copy with SQLite `integrity_check`. Existing rows are preserved. Backups contain the same sensitive student and biometric information as the database, so protect and retain them accordingly.

## Academic setup and use

1. Sign in as the administrator.
2. In Admin Dashboard, add real departments, teachers, and subjects. The application does not create sample academic records. Teacher/subject departments must match.
3. For a single-class rehearsal, open **Get one class ready** in Admin Dashboard. Review the existing department, teacher profile, teacher login, and subject, then save. This links the teacher login and assigns the subject without changing login email or creating sample records. Student and subject membership continues to use the existing department grouping. Setup can be repeated safely.
4. When an administrator creates a student from Students, the student record and login are committed together. SmartAttend generates a unique username from the roll number and a cryptographically random temporary password; the password is displayed once on screen. Give it to the student privately. For an older student without a login, use **Create login** in the roster; use **Reset login** if one-time credentials were lost. Editing a student never creates another account.
5. In Students, create a student record or select an existing one. Enroll with 30 webcam frames or 30 JPEG, PNG, or WebP images. The server accepts enrollment only when at least 10 frames pass image, face, alignment, quality, and ArcFace validation. The roster shows the number of usable stored vectors. Raw images are not retained.
6. Open Take Attendance. A teacher with one assigned subject sees it selected automatically; teachers with several choices can select one. Administrators also select the session teacher. Start attendance and allow camera access. A check-in appears only after the record is committed. Finish Attendance to close the session and finalize it for analytics.
7. The student signs in at `/login` using the username or email and temporary password. First sign-in is limited to Profile & password until they set a private password. They are then redirected to their own Student Portal.
8. If needed, staff can enter or correct attendance manually during a session. The method is labelled `manual` or `manual-correction`, and the acting account, old/new status, time, and note are stored in the audit history. A missed face detection never writes an absent row.
9. Review records with date, subject, student, session, and status filters; download the filtered CSV. Analytics and dashboards read the same database rows. The student portal refreshes saved attendance every 15 seconds and supports date and subject filters.

## Attendance calculation and defaulters

`SMARTATTEND_ATTENDANCE_THRESHOLD_PERCENT` sets the defaulter cutoff (default `75`, range `0`–`100`). The attendance percentage denominator is the count of completed sessions in the student's department; a closed session is treated as finalized, while active sessions are excluded. Empty sessions can be cancelled from session history; they remain visible there and are excluded from attendance calculations. A session with any attendance record cannot be cancelled. Present records count in the numerator. A student with no check-in at a finalized session contributes no present session, but SmartAttend does not create an absent record unless a staff member explicitly enters one. Subject staff analytics divide present entries by eligible student-session opportunities. Because this schema does not track subject registrations, it treats every student in a department as eligible for each session in that department. Percentages are `—` until there is a completed-session denominator. This is a project policy choice; configure it to match the institution before relying on defaulter reports.

## Demonstration preparation

Use the existing database and academic records when they are suitable. Do not seed sample attendance into the regular database. If you need a clean rehearsal, use a separate SQLite file by setting `SMARTATTEND_DATABASE_URL` to a disposable path before initialization and launch. The demo database remains isolated from `smartattend.db`.

The current database has 4 users, 3 departments, 2 teacher profiles, 3 subjects, 4 students, 7 sessions (6 completed and 1 active), 1 saved attendance record, and 1 face-embedding row. The Computer Science department is the existing class grouping with 1 teacher profile, 1 subject, 3 students, and 5 saved sessions; those three students currently have no current face enrollment. Its one active session is empty. After linking the existing teacher login in Class setup, review that session in Take Attendance: close it if it was a real class, or cancel it only if it never took place. Enroll a real Computer Science student before demonstrating live recognition. The existing embedding belongs to another department and is not moved.

Suggested real demonstration sequence:

1. Start SmartAttend with the command above and sign in as the existing administrator (or use first-admin setup on a genuinely empty installation).
2. Open **Get one class ready** and link the existing Computer Science teacher profile to the correct existing teacher login, then assign its subject. Class setup displays roster, enrollment, and active-session counts before saving.
3. In Students, choose one of the real Computer Science students. Enroll their actual face. Use **Create login** if they have no portal account and privately share the one-time credentials.
4. Review the empty active session in Take Attendance. Cancel it only if it never took place; otherwise close it as a real class.
5. Sign in as the linked teacher. The sole assigned subject is selected automatically. Start attendance, present the enrolled student to the webcam, confirm the saved check-in, and finish the session.
6. Open the student's portal in a separate browser session and confirm the committed record appears after automatic refresh. Review the same record in Attendance Records.

Camera troubleshooting: browser camera access needs `localhost` or HTTPS. Allow camera access in the browser's site settings, close apps that already hold the camera, and reload the page. If the camera is unavailable, face enrollment can use 30 supported image files. Live attendance still requires a camera. Do not use screenshots or sample data to claim live recognition.

Browser camera access requires `localhost` or HTTPS and a camera permission grant. The app rejects frames with no face or multiple faces. Unknown or ambiguous matches do not create attendance. The attendance calculation uses completed sessions as the finalized denominator; an absent attendance row is only created through an explicit staff manual entry.

## Biometric data and matching

Web enrollment stores one row per student in `face_embeddings`: contiguous little-endian float32 vectors with 512 dimensions, plus model name, DeepFace version, vector dimension, and JSON preprocessing metadata. Each accepted capture retains its own vector; values are not averaged. Raw webcam images are not stored. Attendance records and session times are stored in SQLite. New timestamps use UTC-naive values to remain compatible with SQLite's existing `DATETIME` columns; date filters also use UTC.

`models/embeddings.pkl` is a separate legacy artifact generated by `src/embedding_engine.py`; that script writes a Python dictionary with `embeddings` and `names`. The web application does not read it. Its pickle format is not imported or overwritten by the web enrollment/recognition workflow.

The installed DeepFace 0.0.99 package declares ArcFace cosine distance threshold `0.68`. SmartAttend uses a stricter `0.20` cutoff and `0.03` runner-up margin, configurable with `SMARTATTEND_FACE_MAX_COSINE_DISTANCE` and `SMARTATTEND_FACE_MIN_MARGIN`. This setting was evaluated on the existing four labeled folders (86 quality-passing images): leave-one-image-out accepted 82/86 known captures with zero wrong known matches; leave-one-identity-out produced zero false accepts among 86 held-out samples. At `0.25`, the same open-set check produced 16 false accepts. This dataset is small and may contain correlated or non-independent captures; it is not a production accuracy or demographic evaluation. The methodology and counts are in [docs/face_matching_calibration.md](docs/face_matching_calibration.md); reproduce it with `venv/bin/python scripts/evaluate_face_threshold.py`.

Face detection is not liveness detection. Use consented, varied classroom samples to recalibrate before consequential use; do not interpret the development-set numbers as an accuracy guarantee.

## Tests

```bash
venv/bin/python -m unittest discover -s tests -v
```

Tests use isolated SQLite databases. The suite covers page and API requests, real-image ArcFace enrollment and recognition, unknown/duplicate faces, persisted check-in-to-dashboard flow, academic/student create and edit, linked accounts, manual attendance audit, CSV filters, student privacy scoping, first-admin setup, profile password changes, verified pre-migration backups, and concurrent duplicate prevention.
