import sys
from pathlib import Path

import cv2
import numpy as np


# ==============================================================
# PROJECT ROOT
# ==============================================================
# enrollment.py is located at:
#
# attendance_ai_project/
# └── backend/
#     └── ai/
#         └── enrollment.py
#
# parents[0] = ai
# parents[1] = backend
# parents[2] = attendance_ai_project
#
# We add the project root to Python's import path so that:
#
# from backend.ai.alignment import FaceAligner
#
# works even when this file is executed directly.
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.ai.alignment import FaceAligner


class FaceEnrollment:

    # ==========================================================
    # CONFIGURATION
    # ==========================================================

    BLUR_THRESHOLD = 35.0

    MIN_BRIGHTNESS = 45.0
    MAX_BRIGHTNESS = 210.0

    MIN_FACE_WIDTH = 100
    MIN_FACE_HEIGHT = 100

    MIN_CONTRAST = 20.0

    NORMALIZED_FACE_SIZE = (200, 200)

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(self):

        # ------------------------------------------------------
        # Face detector
        # ------------------------------------------------------

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.face_detector = cv2.CascadeClassifier(
            cascade_path
        )

        if self.face_detector.empty():

            raise RuntimeError(
                "Could not load OpenCV face detector."
            )

        # ------------------------------------------------------
        # Face aligner
        # ------------------------------------------------------

        self.face_aligner = FaceAligner()

    # ==========================================================
    # FACE DETECTION
    # ==========================================================

    def detect_faces(self, frame):
        """
        Detect faces in the frame.

        Returns a list of bounding boxes.
        """

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Small blur helps reduce camera noise.
        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        # Keep the established stricter detector result when available; a
        # relaxed second pass recovers usable frontal images it used to miss.
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6,
            minSize=(self.MIN_FACE_WIDTH, self.MIN_FACE_HEIGHT),
        )
        if len(faces) == 0:
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60),
            )

        return [
            {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
            for x, y, w, h in faces
        ]

    def detect_face(self, frame):
        """Return the largest detected face for existing callers."""
        faces = self.detect_faces(frame)
        if not faces:
            return None
        return max(faces, key=lambda face: face["width"] * face["height"])

    # ==========================================================
    # FACE CROP
    # ==========================================================

    def crop_face(self, frame, bbox):
        """
        Crop the detected face.

        Adds a small margin around the face.
        """

        x = bbox["x"]
        y = bbox["y"]
        w = bbox["width"]
        h = bbox["height"]

        frame_height, frame_width, _ = frame.shape

        margin_x = int(w * 0.10)
        margin_y = int(h * 0.10)

        x1 = max(
            0,
            x - margin_x
        )

        y1 = max(
            0,
            y - margin_y
        )

        x2 = min(
            frame_width,
            x + w + margin_x
        )

        y2 = min(
            frame_height,
            y + h + margin_y
        )

        face = frame[
            y1:y2,
            x1:x2
        ]

        if face.size == 0:
            return None

        return face

    # ==========================================================
    # NORMALIZE FACE
    # ==========================================================

    def normalize_face(self, face_image):
        """
        Resize the aligned face to 200 × 200.
        """

        return cv2.resize(
            face_image,
            self.NORMALIZED_FACE_SIZE,
            interpolation=cv2.INTER_AREA
        )

    # ==========================================================
    # BLUR CHECK
    # ==========================================================

    def check_blur(self, face_image):
        """
        Use variance of Laplacian to measure sharpness.

        Higher = sharper.
        Lower = blurrier.
        """

        gray = cv2.cvtColor(
            face_image,
            cv2.COLOR_BGR2GRAY
        )

        variance = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        passed = (
            variance >= self.BLUR_THRESHOLD
        )

        return {
            "passed": passed,
            "variance": float(variance)
        }

    # ==========================================================
    # BRIGHTNESS CHECK
    # ==========================================================

    def check_brightness(self, face_image):

        gray = cv2.cvtColor(
            face_image,
            cv2.COLOR_BGR2GRAY
        )

        brightness = float(
            np.mean(gray)
        )

        passed = (
            self.MIN_BRIGHTNESS
            <= brightness
            <= self.MAX_BRIGHTNESS
        )

        return {
            "passed": passed,
            "brightness": brightness
        }

    # ==========================================================
    # CONTRAST CHECK
    # ==========================================================

    def check_contrast(self, face_image):

        gray = cv2.cvtColor(
            face_image,
            cv2.COLOR_BGR2GRAY
        )

        contrast = float(
            np.std(gray)
        )

        passed = (
            contrast >= self.MIN_CONTRAST
        )

        return {
            "passed": passed,
            "contrast": contrast
        }

    # ==========================================================
    # FACE SIZE CHECK
    # ==========================================================

    def check_face_size(self, bbox):

        width = bbox["width"]
        height = bbox["height"]

        passed = (
            width >= self.MIN_FACE_WIDTH
            and
            height >= self.MIN_FACE_HEIGHT
        )

        return {
            "passed": passed,
            "width": width,
            "height": height
        }

    # ==========================================================
    # COMPLETE FRAME PROCESSING
    # ==========================================================

    def process_frame(self, frame):
        """
        Complete SmartAttend enrollment pipeline.

        Camera frame
             ↓
        Face detection
             ↓
        Face size check
             ↓
        Face crop
             ↓
        Face alignment
             ↓
        200 × 200 normalization
             ↓
        Blur check
             ↓
        Brightness check
             ↓
        Contrast check
             ↓
        PASS / FAIL
        """

        # ------------------------------------------------------
        # 1. Detect face
        # ------------------------------------------------------

        detected_faces = self.detect_faces(frame)

        if not detected_faces:

            return {
                "success": False,
                "message": "No face detected"
            }

        if len(detected_faces) > 1:
            return {
                "success": False,
                "message": "More than one face detected. Keep only one face in the frame.",
            }

        bbox = detected_faces[0]

        # ------------------------------------------------------
        # 2. Face size
        # ------------------------------------------------------

        size_result = self.check_face_size(
            bbox
        )

        if not size_result["passed"]:

            return {
                "success": False,
                "message": "Move closer to the camera",
                "bbox": bbox,
                "face_width": size_result["width"],
                "face_height": size_result["height"]
            }

        # ------------------------------------------------------
        # 3. Crop face
        # ------------------------------------------------------

        face_image = self.crop_face(
            frame,
            bbox
        )

        if face_image is None:

            return {
                "success": False,
                "message": "Could not crop face",
                "bbox": bbox
            }

        # ------------------------------------------------------
        # 4. ALIGN FACE
        # ------------------------------------------------------

        aligned_face, angle, eyes = (
            self.face_aligner.align_face(
                face_image
            )
        )

        if aligned_face is None:

            if angle is None:

                message = (
                    "Could not detect both eyes"
                )

            else:

                message = (
                    f"Could not reliably align face "
                    f"(angle={angle:.1f} deg)"
                )

            return {
                "success": False,
                "message": message,
                "bbox": bbox,
                "alignment_angle": angle
            }

        # ------------------------------------------------------
        # 5. Normalize
        # ------------------------------------------------------

        face_image = self.normalize_face(
            aligned_face
        )

        # ------------------------------------------------------
        # 6. Blur
        # ------------------------------------------------------

        blur_result = self.check_blur(
            face_image
        )

        if not blur_result["passed"]:

            return {
                "success": False,
                "message": "Image is too blurry",
                "bbox": bbox,
                "blur_variance": (
                    blur_result["variance"]
                ),
                "alignment_angle": angle
            }

        # ------------------------------------------------------
        # 7. Brightness
        # ------------------------------------------------------

        brightness_result = (
            self.check_brightness(
                face_image
            )
        )

        if not brightness_result["passed"]:

            brightness = (
                brightness_result["brightness"]
            )

            if brightness < self.MIN_BRIGHTNESS:

                message = "Face is too dark"

            else:

                message = "Face is too bright"

            return {
                "success": False,
                "message": message,
                "bbox": bbox,
                "blur_variance": (
                    blur_result["variance"]
                ),
                "brightness": brightness,
                "alignment_angle": angle
            }

        # ------------------------------------------------------
        # 8. Contrast
        # ------------------------------------------------------

        contrast_result = (
            self.check_contrast(
                face_image
            )
        )

        if not contrast_result["passed"]:

            return {
                "success": False,
                "message": (
                    "Image has insufficient contrast"
                ),
                "bbox": bbox,
                "blur_variance": (
                    blur_result["variance"]
                ),
                "brightness": (
                    brightness_result["brightness"]
                ),
                "contrast": (
                    contrast_result["contrast"]
                ),
                "alignment_angle": angle
            }

        # ------------------------------------------------------
        # 9. SUCCESS
        # ------------------------------------------------------

        return {
            "success": True,
            "message": "Face quality is acceptable",
            "bbox": bbox,
            "face": face_image,
            "blur_variance": (
                blur_result["variance"]
            ),
            "brightness": (
                brightness_result["brightness"]
            ),
            "contrast": (
                contrast_result["contrast"]
            ),
            "alignment_angle": angle,
            "eyes": eyes
        }


# ==============================================================
# CAMERA TEST
# ==============================================================

if __name__ == "__main__":

    print()
    print("==============================================")
    print(" SmartAttend Face Enrollment Test")
    print("==============================================")
    print()
    print("Pipeline:")
    print()
    print("Face Detection")
    print("      ↓")
    print("Face Crop")
    print("      ↓")
    print("Face Alignment")
    print("      ↓")
    print("200 × 200 Normalization")
    print("      ↓")
    print("Blur / Brightness / Contrast")
    print("      ↓")
    print("PASS / FAIL")
    print()
    print("Press Q to quit.")
    print()

    # ----------------------------------------------------------
    # Open camera
    # ----------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open camera.")
        exit()

    # Try to use a good webcam resolution.
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    enrollment = FaceEnrollment()

    # ----------------------------------------------------------
    # Camera loop
    # ----------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Could not read camera."
            )

            break

        # ------------------------------------------------------
        # Process frame
        # ------------------------------------------------------

        result = enrollment.process_frame(
            frame
        )

        # ------------------------------------------------------
        # Draw face box
        # ------------------------------------------------------

        if "bbox" in result:

            bbox = result["bbox"]

            x = bbox["x"]
            y = bbox["y"]
            w = bbox["width"]
            h = bbox["height"]

            if result["success"]:

                color = (0, 255, 0)

            else:

                color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                3
            )

        # ------------------------------------------------------
        # Status
        # ------------------------------------------------------

        if result["success"]:

            cv2.putText(
                frame,
                "PASS - READY FOR ENROLLMENT",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                result["message"],
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )

        # ------------------------------------------------------
        # Quality information
        # ------------------------------------------------------

        if "blur_variance" in result:

            cv2.putText(
                frame,
                f"Blur: "
                f"{result['blur_variance']:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        if "brightness" in result:

            cv2.putText(
                frame,
                f"Brightness: "
                f"{result['brightness']:.1f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        if "contrast" in result:

            cv2.putText(
                frame,
                f"Contrast: "
                f"{result['contrast']:.1f}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        if "alignment_angle" in result:

            angle = result["alignment_angle"]

            if angle is not None:

                cv2.putText(
                    frame,
                    f"Alignment: "
                    f"{angle:.1f} deg",
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        # ------------------------------------------------------
        # Show camera
        # ------------------------------------------------------

        cv2.imshow(
            "SmartAttend Enrollment",
            frame
        )

        # ------------------------------------------------------
        # Show final aligned face
        # ------------------------------------------------------

        if result["success"]:

            aligned = result["face"].copy()

            cv2.putText(
                aligned,
                "READY",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "SmartAttend - Normalized Face",
                aligned
            )

        else:

            # Don't show stale face data.
            cv2.destroyWindow(
                "SmartAttend - Normalized Face"
            )

        # ------------------------------------------------------
        # Keyboard
        # ------------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    # ----------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()

    print()
    print("Enrollment test stopped.")
