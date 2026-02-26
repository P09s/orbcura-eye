# 🦯 Orbcura (Vision & Kinematics Engine)

**Part of a Published Patent for an Offline Accessibility Data Workflow**

This repository (`orbcura-eye`) contains the core Computer Vision and full-body kinematic analysis engine for **Orbcura**, an assistive application designed for visually impaired users. 

While the full Orbcura ecosystem includes a Flutter frontend and a USSD-based UPI layer for offline digital payments, this specific module handles the **real-time environment recognition and human action tracking** using edge-optimized AI.

---

## 🧠 How It Works: The Logic & Model

Instead of relying on heavy, cloud-based action recognition models, this engine processes full-body skeletal tracking locally. It uses real-time spatial calculations to determine what a person is doing based on the geometric relationship between their joints.

### 1. Modern Pose Landmark Extraction
The system utilizes the new **MediaPipe Tasks API (Pose Landmarker Lite)**. OpenCV captures the live video feed, converts the color space for AI compatibility, and feeds it into the MediaPipe engine with precise frame timestamps. The model extracts 33 3D skeletal landmarks.

### 2. Skeletal Kinematics & Joint Analysis
The engine isolates specific high-value joints using their standard MediaPipe indices (e.g., `0=Nose`, `11=Left_Shoulder`, `15=Left_Wrist`, `7=Left_Ear`). 
It then calculates the **Euclidean distance** and relative positioning between these points frame-by-frame:
* **Proximity Checks:** Using `math.sqrt()`, we check if two joints are interacting. 
* **Extension Checks:** We measure the absolute X-axis and Y-axis differences between wrists and shoulders to determine arm straightness and extension.

### 3. Action Classification Rules
By defining mathematical thresholds, the engine translates raw coordinates into semantic actions. Current supported actions include:
* **Drinking:** Detected if the distance between either wrist and the nose drops below `0.15`.
* **Punching:** Detected if the X-axis extension of the wrist from the shoulder exceeds `0.4` while the Y-axis alignment remains straight (`< 0.15`).
* **On the Phone:** Detected if the wrist comes within a `0.1` distance threshold of the ear.
* **Hands Above Head:** Detected using inverted Y-axis logic (if wrist Y-coordinates are lesser/higher than the nose Y-coordinate).
* **Facepalm:** Detected if the wrist is extremely close (`< 0.05`) to the eye landmarks.

### 4. Audio Engine Integration
Designed for visually impaired users, visual feedback isn't enough. When a specific action threshold is met, the system triggers a native OS text-to-speech alert (e.g., *"Person is drinking"*). A 3-second cooldown timer ensures the audio doesn't spam the user while an action is ongoing.

---

## 🛠️ Tech Stack

* **Python:** Core programming language.
* **OpenCV (`cv2`):** Video stream processing, RGB conversion, and drawing tracking UI over joints.
* **MediaPipe (`vision.PoseLandmarker`):** Lightweight, edge-optimized skeletal tracking.
* **Math Module:** For real-time Euclidean distance calculations.
* **Native OS APIs:** For zero-latency auditory feedback.

---

## 🚀 Installation & Setup

### Prerequisites
Ensure Python is installed, then install the necessary libraries:
```bash
pip install opencv-python mediapipe
