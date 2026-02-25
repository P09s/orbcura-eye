import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
import math

# --- 1. BOOTSTRAP: Auto-Download Pose Landmarker Model ---
model_path = "pose_landmarker_lite.task"
url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

if not os.path.exists(model_path):
    print("Downloading MediaPipe Pose Model... (This might take a few seconds)")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, model_path)

# --- 2. INITIALIZE MEDIAPIPE TASKS API (The Modern Way) ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.PoseLandmarker.create_from_options(options)

# --- 3. HARDWARE & AUDIO SETUP ---
cap = cv2.VideoCapture(0)
last_spoken_time = 0
cooldown = 3
last_action = ""

print("Modern Action Tracker Booted. Press 'q' to safely quit.")

def calculate_distance(point1, point2):
    return math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)

# Start time for calculating accurate video timestamps
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break

    # Convert OpenCV BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create a native MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # The new API requires a strict millisecond timestamp for every frame to calculate movement
    frame_timestamp_ms = int((time.time() - start_time) * 1000)

    # Detect poses
    results = detector.detect_for_video(mp_image, frame_timestamp_ms)
    
    current_action = None

    # --- 4. ANALYZE THE SKELETON ---
    # The new API can track multiple people, so it returns a list of poses
    if results.pose_landmarks and len(results.pose_landmarks) > 0:
        
        # We only care about the first person it sees [0]
        landmarks = results.pose_landmarks[0]
        
        # Standard MediaPipe Joint Indices: 0=Nose, 11=L_Shoulder, 12=R_Shoulder, 15=L_Wrist, 16=R_Wrist
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        # Draw glowing green circles over the specific joints we are analyzing
        h, w, _ = frame.shape
        for node in [nose, left_shoulder, right_shoulder, left_wrist, right_wrist]:
            cv2.circle(frame, (int(node.x * w), int(node.y * h)), 8, (0, 255, 0), -1)

        # -- Logic A: Detect Drinking --
        dist_left_drink = calculate_distance(left_wrist, nose)
        dist_right_drink = calculate_distance(right_wrist, nose)
        
        if dist_left_drink < 0.15 or dist_right_drink < 0.15:
            current_action = "Person is drinking"

        # -- Logic B: Detect Punching --
        left_punch_extension = abs(left_wrist.x - left_shoulder.x)
        right_punch_extension = abs(right_wrist.x - right_shoulder.x)
        
        left_arm_straight = abs(left_wrist.y - left_shoulder.y) < 0.15
        right_arm_straight = abs(right_wrist.y - right_shoulder.y) < 0.15

        if (left_punch_extension > 0.4 and left_arm_straight) or (right_punch_extension > 0.4 and right_arm_straight):
            current_action = "Person is punching"

        # Y-axis is inverted: lower value = higher on screen
        if left_wrist.y < nose.y and right_wrist.y < nose.y:
            current_action = "Hands above head"

        left_ear = landmarks[7]
        right_ear = landmarks[8]
        
        dist_phone_left = calculate_distance(left_wrist, left_ear)
        dist_phone_right = calculate_distance(right_wrist, right_ear)
        
        if dist_phone_left < 0.1 or dist_phone_right < 0.1:
            current_action = "on the phone"

        left_eye = landmarks[2]
        
        if calculate_distance(left_wrist, left_eye) < 0.05:
            current_action = "facepalm"


    # --- 5. TRIGGER AUDIO ENGINE ---
    if current_action:
        cv2.putText(frame, f"ACTION: {current_action.upper()}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        current_time = time.time()
        
        if (current_time - last_spoken_time > cooldown) or (current_action != last_action):
            os.system(f"say '{current_action}' &")
            last_spoken_time = current_time
            last_action = current_action

    # Render the frame
    cv2.imshow("Orbcura Kinematics (Tasks API)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()