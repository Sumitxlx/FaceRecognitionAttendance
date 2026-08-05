# 🤖 Face Recognition Attendance System

An AI-powered attendance management system that uses **Computer Vision** and **Deep Learning** to automatically recognize faces, mark attendance, and identify defaulters.

---

## 📌 Overview

Traditional attendance systems are time-consuming and prone to proxy attendance. This project automates attendance using facial recognition technology, making the process faster, more accurate, and secure.

The system captures student faces, generates unique face embeddings using **ArcFace**, recognizes individuals in real time, and automatically records attendance.

---

## ✨ Features

- 📸 Student Face Registration
- 👤 Face Detection using OpenCV
- 🧠 Face Recognition using DeepFace + ArcFace
- 🔢 Face Embedding Generation
- ✅ Automatic Attendance Marking
- 📊 Defaulter Identification
- ⚡ Fast and Contactless Attendance System

---

## 🛠️ Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV
- **Deep Learning:** DeepFace
- **Recognition Model:** ArcFace
- **Libraries:** NumPy, Pickle
- **IDE:** Visual Studio Code

---

## 📂 Project Structure

```text
FaceRecognitionAttendance/
│
├── src/
│   ├── register_student.py
│   └── embedding_engine.py
│
├── dataset/
│
├── models/
│
├── train_model.py
│
├── README.md
│
└── .gitignore
```

---

## ⚙️ How It Works

```text
Student
     │
     ▼
 Webcam
     │
     ▼
 Face Detection
     │
     ▼
 Face Alignment
     │
     ▼
 Embedding Generation
     │
     ▼
 Face Matching
     │
     ▼
 Attendance Marked
     │
     ▼
 Defaulter Analysis
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/FaceRecognitionAttendance.git
```

Go to the project directory:

```bash
cd FaceRecognitionAttendance
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Register a student:

```bash
python src/register_student.py
```

Generate face embeddings:

```bash
python src/embedding_engine.py
```

Run attendance:

```bash
python train_model.py
```

---

## 📊 Future Improvements

- Web Dashboard
- Student Login Portal
- Teacher Dashboard
- Admin Panel
- Attendance Analytics
- Liveness Detection
- Cloud Deployment
- Mobile Application

---

## 📸 Screenshots

> Screenshots will be added soon.

---

## 👨‍💻 Author

**Sumit Yadav**

If you like this project, consider giving it a ⭐ on GitHub!
