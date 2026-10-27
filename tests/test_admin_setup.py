import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AdminSetupTests(unittest.TestCase):
    def test_admin_bootstrap_is_first_user_only_and_hashes_password(self):
        with tempfile.TemporaryDirectory(prefix="smartattend-admin-test-") as folder:
            database_path = Path(folder) / "isolated.sqlite"
            environment = os.environ.copy()
            environment["SMARTATTEND_DATABASE_URL"] = f"sqlite:///{database_path}"

            initialized = subprocess.run(
                [sys.executable, "-m", "backend.init_db"],
                cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
            )
            self.assertEqual(initialized.returncode, 0, initialized.stderr)

            created = subprocess.run(
                [sys.executable, "-m", "backend.create_admin"],
                cwd=ROOT, env=environment,
                input="admin@example.com\nstrong test password\nstrong test password\n",
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            self.assertIn("Administrator account created", created.stdout)
            self.assertNotIn("strong test password", created.stdout + created.stderr)

            with sqlite3.connect(database_path) as connection:
                email, password_hash, role = connection.execute(
                    "select email, password_hash, role from users"
                ).fetchone()
            self.assertEqual(email, "admin@example.com")
            self.assertEqual(role, "admin")
            self.assertNotEqual(password_hash, "strong test password")

            second = subprocess.run(
                [sys.executable, "-m", "backend.create_admin"],
                cwd=ROOT, env=environment,
                input="second@example.com\nanother test password\nanother test password\n",
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("bootstrap is disabled", second.stderr)


if __name__ == "__main__":
    unittest.main()
