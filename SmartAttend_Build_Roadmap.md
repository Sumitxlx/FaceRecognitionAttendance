# SmartAttend — Step-by-Step Build Roadmap
### Privacy-First Face Recognition Attendance with Predictive Defaulter Analytics

**Stack:** HTML/CSS/JS (frontend) + FastAPI (backend API) + SQLAlchemy + SQLite→PostgreSQL + DeepFace/ArcFace/OpenCV (AI) + JWT auth + Chart.js

**Golden rule:** at the end of every stage, something new must actually *run* and be *demoable* — even if ugly. Never move to the next stage with a broken previous one. That's what makes the final thing feel "connected" instead of ten disconnected parts glued together at the last minute.

---

## STAGE 0 — Environment & Project Skeleton (Day 1-2)

**Build:**
- Set up folder structure:
  ```
  smartattend/
  ├── backend/
  │   ├── main.py
  │   ├── models.py
  │   ├── database.py
  │   ├── ai/
  │   └── requirements.txt
  ├── frontend/
  │   ├── index.html
  │   ├── css/
  │   └── js/
  └── dataset/        (gitignored)
  ```
- Create a Python virtual environment, install FastAPI + Uvicorn.
- Get `uvicorn main:app --reload` running and returning `{"status": "SmartAttend API running"}` on `/`.

**Done when:** you can open `localhost:8000` in a browser and see that JSON response, and `localhost:8000/docs` shows the auto-generated Swagger page.

---

## STAGE 1 — Static Frontend Shell (Day 2-4)

Build the pages **as static, unstyled HTML first** — no login logic, no data, just structure. This is deliberate: get the skeleton of every screen you'll need before wiring anything up, so later stages just "fill in" pages instead of designing + coding at the same time.

**Pages to create (empty/dummy content is fine):**
- `login.html`
- `admin-dashboard.html`
- `teacher-dashboard.html`
- `student-dashboard.html`
- `take-attendance.html` (this is your centerpiece page)
- `students.html`, `attendance-records.html`, `analytics.html`

**Done when:** you can click between all pages via plain `<a>` links and every page loads (even if blank/ugly).

---

## STAGE 2 — Styling Pass (Day 4-6)

Now make it look like a real product, not a college assignment.

- Set up one shared `style.css` (or Tailwind via CDN) with a consistent color palette, font, and spacing scale — decide this *once*, reuse everywhere.
- Style the login page and one dashboard fully as your "template" — then copy that structure to the other dashboards.
- Build a shared navbar/sidebar component (even if it's just repeated HTML across pages for now).

**Done when:** the login page and at least one dashboard look genuinely presentable — this is your visual anchor for everything else.

---

## STAGE 3 — Database Models (Day 6-8)

Back to the backend. Build your data layer before any logic touches it.

- Write `models.py` with SQLAlchemy models: `Department`, `Teacher`, `Student`, `Subject`, `AttendanceSession`, `AttendanceRecord`, `FaceEmbedding`, `WeeklyAttendanceSnapshot`.
- Set up `database.py` with the SQLite engine + session.
- Write a small `seed.py` script that inserts 1 department, 1 teacher, 1 subject, and 2-3 dummy students — so you always have test data to work against.

**Done when:** running `seed.py` populates `smartattend.db`, and you can open it in a SQLite viewer (e.g. DB Browser for SQLite) and see real rows.

---

## STAGE 4 — Auth (JWT) (Day 8-10)

- Build `/api/register` and `/api/login` endpoints. Passwords hashed with `passlib`/`bcrypt`.
- On login, return a JWT containing user id + role.
- Add a dependency (`get_current_user`) that other routes will use to check the token and enforce role access.
- On the frontend: `login.js` posts credentials, stores the JWT in `localStorage`, redirects based on role.

**Done when:** you can register a test user via Swagger docs, log in via your actual `login.html` page, and see the JWT saved in browser dev tools → Application → Local Storage. Protect one dummy route and confirm it rejects requests without a valid token.

---

## STAGE 5 — Face Enrollment: Capture + Detection + Alignment (Day 10-15)

This is where your AI pipeline plugs in for the first time — keep it as a **standalone module** (`backend/ai/enrollment.py`), separate from your API routes, so it's independently testable.

**Order within this stage matters:**
1. **Face detection first** — use OpenCV/MediaPipe to detect a face in a frame and draw a bounding box. Test this alone with a webcam script before touching FastAPI.
2. **Quality/blur check** — reject blurry or poorly-lit captures (Laplacian variance check is a simple, explainable method — good viva talking point).
3. **Face alignment** — rotate/crop the face based on eye landmarks so all stored faces are consistently oriented before embedding. This measurably improves matching accuracy and is worth explaining clearly in your report.
4. **Multi-image capture** — capture ~30 aligned, quality-passed images per student.

**Then connect to the web:**
- Frontend: `enroll.html` uses `getUserMedia()` to access the webcam, captures frames every ~500ms, POSTs each to `/api/enroll/capture`.
- Backend: runs detection → quality check → alignment on each frame, returns pass/fail + live feedback (e.g. "move closer", "too blurry") as JSON, which your JS displays instantly.

**Done when:** you can open `enroll.html`, look into your webcam, and watch it visibly collect quality-checked, aligned face captures with real-time feedback on screen — this is your first genuinely impressive demo moment.

---

## STAGE 6 — Embedding Generation + Privacy-First Storage (Day 15-18)

- Feed the aligned captures from Stage 5 into DeepFace/ArcFace to generate embeddings.
- Average or store multiple embeddings per student for robustness.
- **Encrypt the embedding vector before saving** to the `FaceEmbedding` table (e.g. with `cryptography.fernet`) — this is your privacy-first differentiator, implemented for real, not just claimed.
- **Delete the raw captured images** after embeddings are successfully generated — write this as an explicit, logged step, not a silent side effect, so you can point to it directly in your report.

**Done when:** after enrolling a student, you can inspect the `dataset/` folder and confirm no raw images remain, and inspect the DB and confirm the stored embedding is encrypted (unreadable without your key).

---

## STAGE 7 — Real-Time Recognition + Matching (Day 18-22)

- Build `backend/ai/recognizer.py`: given a new frame, run detection → alignment → embedding → compare against all stored (decrypted in-memory) embeddings using cosine similarity.
- Set a similarity threshold for "match" vs "unknown."
- Return the matched student id + a **confidence score** — this confidence score is what powers your fallback feature later, so make sure it's part of the response from day one, not bolted on afterward.

**Done when:** you can point your webcam at yourself (already enrolled) and a test script prints your correct identity + confidence score; pointing it at someone unenrolled correctly prints "unknown."

---

## STAGE 8 — Liveness Detection Layer (Day 22-26)

Insert this **before** recognition is trusted, not after:

```
Face Detected → Liveness Check → Real Person? → Recognition → Attendance
```

- Realistic, buildable approach for a student project: **passive texture-based liveness** (checking for print/screen artifacts) combined with a simple **active check** like asking the user to blink or turn their head slightly, detected via eye-aspect-ratio or landmark movement across frames.
- Be explicit in your code/comments and later report that this is a defense layer, not a guarantee — that honesty is a strength, not a weakness, in your viva.

**Done when:** holding a photo of an enrolled student's face up to the camera is correctly rejected as "not live," while the real person passes.

---

## STAGE 9 — Attendance Session Logic (Day 26-30)

Now bring recognition into the actual attendance workflow.

- `/api/session/start` — teacher selects subject/class, creates an `AttendanceSession` row.
- `/api/session/{id}/recognize` — accepts a frame, runs liveness → recognition, and:
  - if confidence is high → mark present (check for duplicates first — one `AttendanceRecord` per student per session, enforced at the query level)
  - if confidence is borderline → flag for manual confirmation instead of guessing (your fallback differentiator)
- `/api/session/{id}/end` — closes the session.

**Done when:** a scripted test (no UI yet) can start a session, "recognize" a known face twice, and confirm only one attendance record is created — duplicate prevention actually works.

---

## STAGE 10 — Wire the Live Attendance UI (Day 30-35)

This is the centerpiece screen — where everything you've built so far becomes one visible flow.

- `take-attendance.html`: teacher clicks "Start Session" → webcam feed appears (via `getUserMedia()`, drawn to a `<canvas>`) → JS captures a frame every second, POSTs to `/api/session/{id}/recognize`.
- On a match: draw a bounding box + name on the canvas overlay, and push the student into a live "Present" list on screen with a checkmark animation.
- On borderline confidence: show a small "confirm this is [Name]?" prompt instead of silently marking.
- Live present-count ticks up as recognitions land.

**Done when:** you can run a full attendance session end-to-end through the actual browser UI — webcam on, faces recognized live, present list populating, duplicates blocked — no scripts, no Swagger docs, just the app working. This is your core demo.

---

## STAGE 11 — Dashboards With Real Data (Day 35-40)

Go back to your Stage 1/2 dashboard pages and connect them to real API data instead of placeholders.

- Teacher dashboard: today's sessions, present/absent counts, recent sessions list.
- Student dashboard: their own attendance %, subject-wise breakdown, history table.
- Admin dashboard: totals across students/teachers/classes, today's attendance.

**Done when:** logging in as each of the three roles shows genuinely different, real, correct data pulled from the database.

---

## STAGE 12 — Analytics + Charts (Day 40-44)

- Build `/api/analytics/...` endpoints returning aggregated data (attendance over time, subject-wise %, department stats).
- Render with Chart.js: line chart for trends, bar chart for subject comparison, pie/doughnut for present-vs-absent.

**Done when:** the analytics page shows real charts that visibly change as you run more test attendance sessions.

---

## STAGE 13 — Predictive At-Risk Model (Day 44-48)

Your key differentiator — build it carefully, it's worth the extra days.

- A scheduled/manual job populates `WeeklyAttendanceSnapshot` (weekly attendance % per student).
- Once you have a few weeks of data, fit a simple linear regression (scikit-learn) per student on their weekly % trend, and project forward.
- Flag students whose *projected* trend crosses below 75% within the next N weeks — even if their *current* % is still safe.
- On the student/teacher dashboard, show a small explanation: a mini trend chart with the flagged weeks highlighted, not just a red "at risk" badge with no reasoning.

**Done when:** you can simulate a student's declining weekly attendance (via your seed/test data) and watch the system flag them as at-risk *before* they actually cross the threshold, with the trend chart visibly showing why.

---

## STAGE 14 — Evaluation (Day 48-50)

This feeds directly into your report/paper's Results section.

- Run your recognition system against your full enrolled dataset and log accuracy, False Accept Rate, and False Reject Rate at 2-3 different similarity thresholds.
- Save this as a small script (`evaluate.py`) that outputs a results table — this becomes both a report table and a nice "we tested this rigorously" viva moment.

**Done when:** you have real numbers, not guesses, for how accurate your system is.

---

## STAGE 15 — Stretch Feature (Day 50-55)

Pick **one**, don't split effort across both:
- **Liveness robustness improvements** (if you want a more technically impressive viva) — add a second liveness signal, test against printed-photo and screen-replay attacks, report results.
- **Email notifications** (if you want a more demo-friendly "wow" moment) — auto-email a student when flagged at-risk, using their trend chart as context.

**Done when:** the feature works end-to-end and you can demo it live.

---

## STAGE 16 — Polish, Testing, Docs, GitHub (Day 55-60)

- Full pass over the UI: consistent spacing, loading states, error messages instead of silent failures.
- Write a proper `README.md`: problem, architecture diagram, setup instructions, screenshots.
- `.gitignore`: exclude `dataset/`, `.db` files, `.env`, `venv/`.
- `requirements.txt` pinned and tested on a fresh virtual environment (confirm it actually installs clean).
- Write your report/paper using the Stage 14 evaluation numbers as your Results section.

**Done when:** a stranger could clone your repo, follow the README, and get the app running without asking you a single question.

---

## How to actually use this roadmap

Work top to bottom. Don't skip ahead to a "cooler" stage because it's more exciting — Stage 10 (the live attendance demo) only feels impressive *because* Stages 3-9 quietly made it possible. Every stage's "Done when" line is your checkpoint: if you can't tick it, don't move on, come back and we'll debug it together.
