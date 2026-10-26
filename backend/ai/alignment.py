import cv2
import numpy as np


class FaceAligner:

    OUTPUT_SIZE = 200

    # Safety limit.
    # If eye detection produces a ridiculous angle,
    # we reject it instead of rotating the face incorrectly.
    MAX_ROTATION_ANGLE = 25.0

    def __init__(self):

        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        # Eye detector
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_eye.xml"
        )

        if self.face_cascade.empty():
            raise RuntimeError(
                "Could not load face Haar cascade."
            )

        if self.eye_cascade.empty():
            raise RuntimeError(
                "Could not load eye Haar cascade."
            )

    # =========================================================
    # FACE DETECTION
    # =========================================================

    def detect_face(self, frame):
        """
        Detect the largest face in the frame.

        Returns:
            (x, y, w, h)
            or None
        """

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100)
        )

        if len(faces) == 0:
            return None

        # Choose the largest detected face
        face = max(
            faces,
            key=lambda rectangle: rectangle[2] * rectangle[3]
        )

        return tuple(face)

    # =========================================================
    # EYE DETECTION
    # =========================================================

    def detect_eyes(self, face):
        """
        Detect two eyes inside a cropped face.

        We only search the upper portion of the face
        to reduce false detections.
        """

        gray = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        height, width = gray.shape

        # Eyes are normally in the upper part of the face.
        upper_face_height = int(height * 0.60)

        upper_face = gray[
            0:upper_face_height,
            0:width
        ]

        eyes = self.eye_cascade.detectMultiScale(
            upper_face,
            scaleFactor=1.08,
            minNeighbors=6,
            minSize=(20, 20),
            maxSize=(
                max(30, int(width * 0.45)),
                max(30, int(height * 0.30))
            )
        )

        if len(eyes) < 2:
            return None

        candidates = []

        for (x, y, w, h) in eyes:

            center_x = x + w / 2
            center_y = y + h / 2

            # Ignore candidates too low in the face.
            if center_y > height * 0.55:
                continue

            # Basic eye shape filtering.
            aspect_ratio = w / float(h)

            if aspect_ratio < 0.5 or aspect_ratio > 3.5:
                continue

            candidates.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "center": (
                        center_x,
                        center_y
                    )
                }
            )

        if len(candidates) < 2:
            return None

        # -----------------------------------------------------
        # Find the best pair of eye candidates.
        # -----------------------------------------------------

        best_pair = None
        best_score = float("inf")

        for i in range(len(candidates)):

            for j in range(i + 1, len(candidates)):

                eye1 = candidates[i]
                eye2 = candidates[j]

                x1, y1 = eye1["center"]
                x2, y2 = eye2["center"]

                # Put left eye first.
                if x1 > x2:

                    eye1, eye2 = eye2, eye1

                    x1, y1 = eye1["center"]
                    x2, y2 = eye2["center"]

                horizontal_distance = x2 - x1

                vertical_difference = abs(y2 - y1)

                # Eyes should have a reasonable separation.
                if horizontal_distance < width * 0.20:
                    continue

                if horizontal_distance > width * 0.85:
                    continue

                # Eyes should be approximately on the same row.
                if vertical_difference > height * 0.15:
                    continue

                # Eye sizes should be somewhat similar.
                size_difference = (
                    abs(eye1["w"] - eye2["w"])
                    / max(
                        eye1["w"],
                        eye2["w"]
                    )
                )

                if size_difference > 0.60:
                    continue

                # Smaller score means better pair.
                score = (
                    vertical_difference * 5
                    + size_difference * 20
                )

                if score < best_score:

                    best_score = score

                    best_pair = (
                        (x1, y1),
                        (x2, y2)
                    )

        return best_pair

    # =========================================================
    # ANGLE
    # =========================================================

    def calculate_angle(self, left_eye, right_eye):
        """
        Calculate the angle between the eyes.

        Horizontal eyes ≈ 0 degrees.
        """

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        angle = np.degrees(
            np.arctan2(dy, dx)
        )

        return float(angle)

    # =========================================================
    # ALIGN FACE
    # =========================================================

    def align_face(self, face):
        """
        Align the face so the eyes become approximately horizontal.

        Returns:

            aligned_face,
            angle,
            eye_points

        or:

            None,
            None,
            None
        """

        eyes = self.detect_eyes(face)

        if eyes is None:
            return None, None, None

        left_eye, right_eye = eyes

        angle = self.calculate_angle(
            left_eye,
            right_eye
        )

        # -----------------------------------------------------
        # Safety check.
        # -----------------------------------------------------

        if abs(angle) > self.MAX_ROTATION_ANGLE:

            return None, angle, eyes

        height, width = face.shape[:2]

        # Rotate around midpoint between the eyes.
        eye_center = (
            int(
                (left_eye[0] + right_eye[0]) / 2
            ),
            int(
                (left_eye[1] + right_eye[1]) / 2
            )
        )

        rotation_matrix = cv2.getRotationMatrix2D(
            eye_center,
            angle,
            1.0
        )

        aligned = cv2.warpAffine(
            face,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        # Center and make square.
        aligned = self.crop_centered_face(
            aligned
        )

        # Standard size for later processing.
        aligned = cv2.resize(
            aligned,
            (
                self.OUTPUT_SIZE,
                self.OUTPUT_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        return aligned, angle, eyes

    # =========================================================
    # CENTERED CROP
    # =========================================================

    def crop_centered_face(self, face):
        """
        Make a centered square crop.
        """

        height, width = face.shape[:2]

        size = min(
            height,
            width
        )

        center_x = width // 2
        center_y = height // 2

        half = size // 2

        x1 = max(
            0,
            center_x - half
        )

        y1 = max(
            0,
            center_y - half
        )

        x2 = min(
            width,
            x1 + size
        )

        y2 = min(
            height,
            y1 + size
        )

        return face[
            y1:y2,
            x1:x2
        ]

    # =========================================================
    # COMPLETE FRAME PROCESSING
    # =========================================================

    def process_frame(self, frame):
        """
        Complete alignment pipeline:

            camera frame
                 ↓
            detect face
                 ↓
            crop face
                 ↓
            detect eyes
                 ↓
            calculate angle
                 ↓
            rotate face
                 ↓
            center face
                 ↓
            200 x 200
        """

        face_box = self.detect_face(
            frame
        )

        if face_box is None:

            return {
                "success": False,
                "reason": "No face detected",
                "aligned_face": None,
                "face_box": None,
                "angle": None,
                "eyes": None
            }

        x, y, w, h = face_box

        # Add a small margin around the detected face.
        margin_x = int(w * 0.12)
        margin_y = int(h * 0.12)

        x1 = max(
            0,
            x - margin_x
        )

        y1 = max(
            0,
            y - margin_y
        )

        x2 = min(
            frame.shape[1],
            x + w + margin_x
        )

        y2 = min(
            frame.shape[0],
            y + h + margin_y
        )

        face = frame[
            y1:y2,
            x1:x2
        ]

        aligned, angle, eyes = self.align_face(
            face
        )

        if aligned is None:

            if angle is None:

                reason = "Eyes not detected"

            else:

                reason = (
                    f"Unreliable eye angle: "
                    f"{angle:.1f} degrees"
                )

            return {
                "success": False,
                "reason": reason,
                "aligned_face": None,
                "face_box": (
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                ),
                "angle": angle,
                "eyes": eyes
            }

        # Convert eye coordinates from cropped-face
        # coordinates back into camera coordinates.
        frame_eyes = None

        if eyes is not None:

            left_eye, right_eye = eyes

            frame_eyes = (
                (
                    int(x1 + left_eye[0]),
                    int(y1 + left_eye[1])
                ),
                (
                    int(x1 + right_eye[0]),
                    int(y1 + right_eye[1])
                )
            )

        return {
            "success": True,
            "reason": "Face aligned",
            "aligned_face": aligned,
            "face_box": (
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ),
            "angle": angle,
            "eyes": frame_eyes
        }


# =============================================================
# CAMERA TEST
# =============================================================

def test_alignment():

    aligner = FaceAligner()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("ERROR: Could not open camera.")

        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    print()
    print("=" * 60)
    print("SMARTATTEND FACE ALIGNMENT TEST")
    print("=" * 60)
    print()
    print("Look at the camera.")
    print("Move your head slightly left/right.")
    print("Try tilting your head.")
    print("Press Q to quit.")
    print()

    while True:

        success, frame = camera.read()

        if not success:

            print("Could not read camera frame.")

            break

        result = aligner.process_frame(
            frame
        )

        # -----------------------------------------------------
        # FACE BOX
        # -----------------------------------------------------

        if result["face_box"] is not None:

            x, y, w, h = result["face_box"]

            if result["success"]:

                box_color = (0, 255, 0)

            else:

                box_color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2
            )

        # -----------------------------------------------------
        # EYES
        # -----------------------------------------------------

        if result["eyes"] is not None:

            left_eye, right_eye = result["eyes"]

            cv2.circle(
                frame,
                left_eye,
                5,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                frame,
                right_eye,
                5,
                (255, 0, 0),
                -1
            )

            cv2.line(
                frame,
                left_eye,
                right_eye,
                (0, 255, 255),
                2
            )

        # -----------------------------------------------------
        # STATUS
        # -----------------------------------------------------

        if result["success"]:

            status = "ALIGNED"
            status_color = (0, 255, 0)

        else:

            status = result["reason"]
            status_color = (0, 0, 255)

        cv2.putText(
            frame,
            status,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        # -----------------------------------------------------
        # ANGLE
        # -----------------------------------------------------

        if result["angle"] is not None:

            cv2.putText(
                frame,
                f"Angle: {result['angle']:.1f} deg",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # -----------------------------------------------------
        # CAMERA
        # -----------------------------------------------------

        cv2.imshow(
            "SmartAttend - Face Alignment",
            frame
        )

        # -----------------------------------------------------
        # ALIGNED FACE
        # -----------------------------------------------------

        if result["aligned_face"] is not None:

            aligned_display = (
                result["aligned_face"].copy()
            )

            cv2.putText(
                aligned_display,
                "ALIGNED",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "SmartAttend - Aligned Face (200x200)",
                aligned_display
            )

        else:

            placeholder = np.zeros(
                (
                    FaceAligner.OUTPUT_SIZE,
                    FaceAligner.OUTPUT_SIZE,
                    3
                ),
                dtype=np.uint8
            )

            cv2.putText(
                placeholder,
                "Waiting...",
                (35, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "SmartAttend - Aligned Face (200x200)",
                placeholder
            )

        # -----------------------------------------------------
        # QUIT
        # -----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    camera.release()

    cv2.destroyAllWindows()

    print()
    print("Face alignment test stopped.")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    test_alignment()