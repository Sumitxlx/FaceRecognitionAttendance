from datetime import datetime, timedelta, timezone
import os
import secrets

from jose import jwt


# Set SMARTATTEND_SECRET_KEY to keep tokens valid across restarts.
# The process-local fallback avoids committing a reusable secret.
SECRET_KEY = os.environ.get("SMARTATTEND_SECRET_KEY") or secrets.token_urlsafe(32)

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 30


def create_access_token(data: dict):
    """
    Create a JWT access token.
    """

    to_encode = data.copy()

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    to_encode.update({
        "exp": expire
    })

    return jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )
