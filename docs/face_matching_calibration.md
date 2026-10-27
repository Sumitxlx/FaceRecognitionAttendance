# ArcFace distance calibration

This records a small development-set calibration. It is not an accuracy or demographic performance claim.

## Configuration

- DeepFace `0.0.99`, model `ArcFace`, 512D float32 vectors.
- Cosine distance: `1 - dot(L2_normalize(probe), L2_normalize(reference))`, matching DeepFace's installed `find_cosine_distance` implementation.
- DeepFace's installed `deepface/config/threshold.py` sets ArcFace cosine distance to `0.68`.
- SmartAttend instead defaults to `0.20` and a best-to-runner-up gap of `0.03`, configurable with `SMARTATTEND_FACE_MAX_COSINE_DISTANCE` and `SMARTATTEND_FACE_MIN_MARGIN`.
- Samples were processed with the same OpenCV face-quality/alignment pipeline and `DeepFace.represent(model_name="ArcFace", detector_backend="skip", align=False)` path used by enrollment and live recognition.

## Existing dataset result

The four labeled directories contained 120 images. The configured quality pipeline accepted 86 (Himanshu 25, Priya 25, Sumit 28, Sumit2 8). Evaluation used nearest-vector per identity, leave-one-image-out for known identities, and held out each entire identity in turn for an open-set check.

| Cosine cutoff | Correct known accepts | Wrong known accepts | Known rejects | Open-set false accepts |
|---:|---:|---:|---:|---:|
| 0.20 | 82/86 | 0 | 4/86 | 0/86 |
| 0.25 | 85/86 | 0 | 1/86 | 16/86 |
| 0.30 | 86/86 | 0 | 0/86 | 30/86 |
| 0.40 | 86/86 | 0 | 0/86 | 34/86 |
| 0.68 (DeepFace default) | 86/86 | 0 | 0/86 | 40/86 |

The low cutoff was chosen to reduce false acceptance on this small open-set check, with four known captures rejected. The directories may not represent independent identities or real classroom conditions, and repeated frames may be correlated. Re-evaluate with consented, varied genuine and non-matching classroom captures before changing the default or relying on attendance decisions. Recognition is not liveness detection.

Reproduce the evaluation with:

```bash
venv/bin/python scripts/evaluate_face_threshold.py
```
