# --- 1. BOOTSTRAP: Auto-Download Raw Model Files ---
import cv2
import numpy as np
import os
import urllib.request
import time

files = {
    "yolov3-tiny.weights": "https://pjreddie.com/media/files/yolov3-tiny.weights",
    "yolov3-tiny.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
}

# Trick the server into thinking this script is a standard Mac web browser
opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')]
urllib.request.install_opener(opener)

print("Verifying neural network files...")
for filename, url in files.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}... (This might take a minute)")
        urllib.request.urlretrieve(url, filename)

# --- 2. INITIALIZE OPENCV'S DNN MODULE ---
# Load the vocabulary (80 common objects including 'person' and 'cell phone')
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the network directly into OpenCV
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
output_layers = net.getUnconnectedOutLayersNames()

# --- 3. HARDWARE & AUDIO SETUP ---
cap = cv2.VideoCapture(0)
last_spoken_time = 0
cooldown = 3 # Wait 3 seconds before speaking again so it doesn't spam your speakers

print("Camera active. Press 'q' to safely quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    height, width, channels = frame.shape

    # --- 4. TENSOR CONVERSION ---
    # We must convert the raw image into a normalized "blob" for the neural network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    # Push the blob through the network
    outs = net.forward(output_layers)

    # --- 5. PROCESS DETECTIONS ---
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # We only care about detections with >50% confidence
            if confidence > 0.5:
                # Map the network's normalized coordinates back to your screen size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Get the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove duplicate bounding boxes overlapping the same object
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []

    # --- 6. DRAW INTERFACE & SPEAK ---
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_objects.append(label)

            # Draw the box and text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Audio Engine Logic
    if detected_objects:
        primary_object = detected_objects[0]
        current_time = time.time()
        
        if current_time - last_spoken_time > cooldown:
            # Native macOS text-to-speech. 
            # The '&' runs it on a separate background thread so your video doesn't freeze!
            os.system(f"say '{primary_object}' &")
            last_spoken_time = current_time

    # Render the frame
    cv2.imshow("Raw OpenCV Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up memory
cap.release()
cv2.destroyAllWindows()