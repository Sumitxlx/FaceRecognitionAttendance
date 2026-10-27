"""Evaluate ArcFace cosine cutoffs on the labeled folders in dataset/."""

from collections import defaultdict
from pathlib import Path
import re
from importlib.metadata import distribution
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ai.recognition import (
    DISTANCE_METRIC,
    MAX_COSINE_DISTANCE,
    MIN_MATCH_MARGIN,
    MODEL_NAME,
    MODEL_VERSION,
    FaceRecognition,
)


def official_arcface_cosine_threshold() -> float | None:
    path = Path(distribution("deepface").locate_file("deepface/config/threshold.py"))
    source = path.read_text()
    match = re.search(r'"ArcFace"\s*:\s*\{\s*"cosine"\s*:\s*([0-9.]+)', source)
    return float(match.group(1)) if match else None


def main() -> None:
    root = PROJECT_ROOT / "dataset"
    recognizer = FaceRecognition()
    gallery: dict[str, list[np.ndarray]] = defaultdict(list)
    counts = {}

    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(
            path for path in folder.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        rejected = 0
        for path in files:
            frame = cv2.imread(str(path))
            if frame is None:
                rejected += 1
                continue
            vector, reason = recognizer.embed_frame(frame)
            if reason or vector is None:
                rejected += 1
                continue
            vector = vector / np.linalg.norm(vector)
            gallery[folder.name].append(vector)
        counts[folder.name] = {"accepted": len(gallery[folder.name]), "rejected": rejected}

    labels = list(gallery)
    cutoffs = sorted({0.20, 0.25, 0.30, 0.35, 0.40, MAX_COSINE_DISTANCE,
                      official_arcface_cosine_threshold() or 0.68})
    print(f"model={MODEL_NAME} version={MODEL_VERSION} metric={DISTANCE_METRIC}")
    print(f"configured_threshold={MAX_COSINE_DISTANCE} margin={MIN_MATCH_MARGIN}")
    print(f"official_package_threshold={official_arcface_cosine_threshold()}")
    print(f"quality_filtered_samples={counts}")

    nearest_genuine = []
    nearest_impostor = []
    for threshold in cutoffs:
        correct = wrong = rejected = open_false_accept = total = 0
        for identity, samples in gallery.items():
            for index, probe in enumerate(samples):
                distances = {}
                for candidate, references in gallery.items():
                    if candidate == identity:
                        references = np.delete(np.asarray(references), index, axis=0)
                    else:
                        references = np.asarray(references)
                    if len(references):
                        distances[candidate] = float(np.min(1.0 - references @ probe))

                own_distance = distances.get(identity, float("inf"))
                other_distances = [value for key, value in distances.items() if key != identity]
                if other_distances:
                    nearest_genuine.append(own_distance)
                    nearest_impostor.append(min(other_distances))
                ranked = sorted(distances.items(), key=lambda item: item[1])
                if len(ranked) > 1:
                    best, second = ranked[:2]
                    accepted = best[1] <= threshold and second[1] - best[1] >= MIN_MATCH_MARGIN
                    if accepted and best[0] == identity:
                        correct += 1
                    elif accepted:
                        wrong += 1
                    else:
                        rejected += 1

                # Open-set evaluation holds out each whole identity in turn.
                unknown = sorted(
                    (value for label, value in distances.items() if label != identity)
                )
                if len(unknown) > 1 and unknown[0] <= threshold and unknown[1] - unknown[0] >= MIN_MATCH_MARGIN:
                    open_false_accept += 1
                total += 1

        print(
            f"threshold={threshold:.2f} closed_correct={correct} "
            f"wrong_known_match={wrong} rejected_known={rejected} "
            f"open_set_false_accept={open_false_accept}/{total}"
        )

    if nearest_genuine and nearest_impostor:
        print("genuine_nearest_min_median_p90_max=", _summary(nearest_genuine))
        print("impostor_nearest_min_median_p10_max=", _summary(nearest_impostor, 0.10))


def _summary(values, quantile=0.90):
    values = np.asarray(values)
    return [round(float(x), 4) for x in (
        values.min(), np.median(values), np.quantile(values, quantile), values.max()
    )]


if __name__ == "__main__":
    main()
