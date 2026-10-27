"""Shared ArcFace preprocessing, embedding, and nearest-identity matching."""

from importlib.metadata import PackageNotFoundError, version
import json
import os

import numpy as np

from backend.ai.enrollment import FaceEnrollment


MODEL_NAME = "ArcFace"
DISTANCE_METRIC = "cosine"
VECTOR_DIMENSION = 512
EMBEDDING_DTYPE = np.dtype("<f4")
MIN_MATCH_MARGIN = float(os.getenv("SMARTATTEND_FACE_MIN_MARGIN", "0.03"))
MAX_COSINE_DISTANCE = float(os.getenv("SMARTATTEND_FACE_MAX_COSINE_DISTANCE", "0.20"))
if not np.isfinite(MAX_COSINE_DISTANCE) or not 0.0 <= MAX_COSINE_DISTANCE <= 2.0:
    raise RuntimeError("SMARTATTEND_FACE_MAX_COSINE_DISTANCE must be between 0 and 2")
if not np.isfinite(MIN_MATCH_MARGIN) or not 0.0 <= MIN_MATCH_MARGIN <= 2.0:
    raise RuntimeError("SMARTATTEND_FACE_MIN_MARGIN must be between 0 and 2")

try:
    MODEL_VERSION = version("deepface")
except PackageNotFoundError:
    MODEL_VERSION = "unknown"

PREPROCESSING_METADATA = json.dumps({
    "face_detector": "OpenCV Haar frontal-face with relaxed fallback",
    "aligner": "backend.ai.alignment.FaceAligner",
    "normalized_face_size": [200, 200],
    "deepface_detector_backend": "skip",
    "deepface_align": False,
    "distance_metric": DISTANCE_METRIC,
}, sort_keys=True, separators=(",", ":"))


def decode_embedding_matrix(blob: bytes) -> np.ndarray | None:
    """Decode contiguous little-endian float32 ArcFace rows and L2-normalize."""
    if not blob or len(blob) % (VECTOR_DIMENSION * EMBEDDING_DTYPE.itemsize):
        return None
    vectors = np.frombuffer(blob, dtype=EMBEDDING_DTYPE)
    matrix = vectors.reshape(-1, VECTOR_DIMENSION)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or np.any(norms <= 1e-12):
        return None
    return matrix / norms


def choose_face_match(probe: np.ndarray, candidates, *, threshold=None, margin=None):
    """Return (candidate, cosine distance, reason); never choose ambiguous/unknown faces."""
    max_distance = MAX_COSINE_DISTANCE if threshold is None else threshold
    min_margin = MIN_MATCH_MARGIN if margin is None else margin
    vector = np.asarray(probe, dtype=EMBEDDING_DTYPE)
    if vector.shape != (VECTOR_DIMENSION,) or not np.isfinite(vector).all():
        return None, None, "Invalid face embedding"
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None, None, "Invalid face embedding"
    vector = vector / norm

    ranked = []
    for candidate, blob in candidates:
        references = decode_embedding_matrix(blob)
        if references is None:
            continue
        distance = float(np.min(1.0 - references @ vector))
        ranked.append((distance, candidate))
    ranked.sort(key=lambda item: item[0])
    if not ranked or ranked[0][0] > max_distance:
        return None, None, "No confident match"
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < min_margin:
        return None, None, "Ambiguous face match"
    return ranked[0][1], ranked[0][0], None


class FaceRecognition:
    """One implementation of face detection, aligned embedding, and matching."""

    def __init__(self):
        self.enrollment = FaceEnrollment()

    def preprocess_frame(self, frame):
        return self.enrollment.process_frame(frame)

    @staticmethod
    def embed_aligned_face(face_image) -> np.ndarray:
        from deepface import DeepFace

        result = DeepFace.represent(
            img_path=face_image,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )
        if not result or "embedding" not in result[0]:
            raise ValueError("ArcFace returned no embedding")
        vector = np.asarray(result[0]["embedding"], dtype=EMBEDDING_DTYPE)
        if (vector.shape != (VECTOR_DIMENSION,) or not np.isfinite(vector).all()
                or np.linalg.norm(vector) <= 1e-12):
            raise ValueError("ArcFace returned an invalid embedding")
        return vector

    def embed_frame(self, frame):
        processed = self.preprocess_frame(frame)
        if not processed.get("success") or processed.get("face") is None:
            return None, processed.get("message", "Face processing failed")
        return self.embed_aligned_face(processed["face"]), None
