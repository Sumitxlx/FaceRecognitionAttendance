import os
import pickle
from deepface import DeepFace

# =================================================
# MODULE 2 : FACE EMBEDDING GENERATION ENGINE
# =================================================
#
# Purpose of this script:
# -----------------------
# Convert all face images inside the dataset folder into
# numerical face representations called **embeddings**.
#
# These embeddings act like a **mathematical fingerprint**
# for each face and will later be used to perform
# real‑time face recognition.
#
# Pipeline implemented here:
#
# dataset images
#        ↓
# face detection + alignment
#        ↓
# ArcFace neural network
#        ↓
# 512‑dimensional embedding vector
#        ↓
# save embeddings.pkl (face database)
#
# =================================================


# =================================================
# PATH SETUP
# =================================================

# __file__ gives the location of this script
# Example:
# /Users/sumit/attendance_ai_project/src/embedding_engine.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the dataset path
# ".." means go one directory up from /src → project root
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset"))

# Folder where trained embeddings will be stored
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

# Create models directory if it does not exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Final output file
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "embeddings.pkl")


# =================================================
# CONFIGURATION
# =================================================

# Face recognition model used by DeepFace
# ArcFace is one of the most accurate face recognition models
MODEL_NAME = "ArcFace"


# =================================================
# STORAGE VARIABLES
# =================================================
# These lists will store the results of the embedding generation

embeddings = []   # stores face vectors
names = []        # stores corresponding person names


print("\n[INFO] Starting embedding generation...\n")


# =================================================
# DATASET PROCESSING LOOP
# =================================================
#
# Dataset structure expected:
#
# dataset/
#     sumit/
#         0.jpg
#         1.jpg
#     anjali/
#         0.jpg
#         1.jpg
#
# Each folder represents a person.

for person_name in os.listdir(DATASET_DIR):

    # Full path to a person's folder
    person_folder = os.path.join(DATASET_DIR, person_name)

    # Skip files that are not folders
    if not os.path.isdir(person_folder):
        continue

    print(f"[INFO] Processing student: {person_name}")

    # Loop through every image of that person
    for image_name in os.listdir(person_folder):

        # Build the full image path
        image_path = os.path.join(person_folder, image_name)

        try:

            # =================================================
            # FACE EMBEDDING GENERATION
            # =================================================
            #
            # DeepFace.represent() performs several steps internally:
            #
            # 1️⃣ Face Detection
            # 2️⃣ Face Alignment
            # 3️⃣ Feature Extraction using ArcFace
            #
            # Output is a vector of ~512 numbers that uniquely
            # represent the face.

            representation = DeepFace.represent(
                img_path=image_path,
                model_name=MODEL_NAME,
                enforce_detection=False
            )

            # Extract the embedding vector from the result
            embedding_vector = representation[0]["embedding"]

            # Store embedding
            embeddings.append(embedding_vector)

            # Store the name associated with this embedding
            names.append(person_name)

        except Exception as e:

            # If any image fails processing (corrupt file etc)
            # the system skips it and continues.
            print(f"[WARNING] Skipping {image_name}: {e}")


# =================================================
# SAVE EMBEDDINGS DATABASE
# =================================================
#
# We save the embeddings using pickle.
# Pickle allows Python objects to be stored in a file.

data = {
    "embeddings": embeddings,
    "names": names
}

# Open file in binary write mode
with open(EMBEDDINGS_PATH, "wb") as f:

    # Serialize the data
    pickle.dump(data, f)


# =================================================
# COMPLETION MESSAGE
# =================================================

print("\n[SUCCESS] Embeddings generated successfully")
print(f"[INFO] Total embeddings stored: {len(embeddings)}")
print(f"[INFO] Saved to: {EMBEDDINGS_PATH}")