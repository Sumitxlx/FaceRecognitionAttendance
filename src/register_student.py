import cv2
import os
import time

# Build dataset path relative to project root (so it works no matter where the script is run from)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
DATASET_DIR = os.path.abspath(DATASET_DIR)
IMAGE_SIZE = 224
MAX_IMAGES = 30
CAPTURE_DELAY = 0.6
BLUR_THRESHOLD = 100
MIN_FACE_SIZE = 120

#create dataset directory if it doesn't exist
os.makedirs(DATASET_DIR,exist_ok=True)

# ask for student name
name = input("Enter student name: ").strip()

#create a folder for this student
student_dir = os.path.join(DATASET_DIR,name)
os.makedirs(student_dir, exist_ok=True)

print("[INFO] Dataset folder:", student_dir)

# Blur Detection Function
def blur_score(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# Face detector

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

#webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Camera not accesible")
    exit()
print("[INFO] Camera Started")
count = 0
last_capture_time = 0


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to access camera")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80,80)
    )

    quality_status = "NO FACE"

    if len(faces) > 0:
        faces = sorted(faces, key= lambda x : x[2]*x[3], reverse=True) #area = width * height and select the first item in reverse order
        (x,y,w,h) = faces[0]
        face = frame[y:y+h,x:x+w]
        face = cv2.resize(face,(IMAGE_SIZE,IMAGE_SIZE))

        blur = blur_score(face)
        if blur < BLUR_THRESHOLD:
            quality_status = "BLURRY"
        elif w < MIN_FACE_SIZE:
            quality_status = "TOO FAR"
        else:
            quality_status = "GOOD"

        #draw rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # capture logic
        if quality_status == "GOOD":
            current_time = time.time()
            if current_time - last_capture_time > CAPTURE_DELAY:
                img_path = os.path.join(student_dir,f"{count}.jpg")
                cv2.imwrite(img_path,face)
                print(f"[DATASET] Saved image {count}")
                count += 1
                last_capture_time = current_time
        
        # show blur score
        cv2.putText(
            frame,
            f"Blur:{int(blur)}",
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,0),
            2
        )

    # Display Information
    cv2.putText(
        frame,
        f"Status: {quality_status}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.imshow("Face Registration System",frame)

    if count>= MAX_IMAGES:
        print("[SUCCESS] Registration completed")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
