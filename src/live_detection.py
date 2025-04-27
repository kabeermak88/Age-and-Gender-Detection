import cv2
import time
import numpy as np
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_skip = 5  # Process every 5th frame for faster updates
frame_count = 0

detections = []  # Stores previous detection results to reduce flickering
detection_time = 3  # Keep results on screen for 3 seconds
last_detection_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    frame_count += 1

    # Process every nth frame
    if frame_count % frame_skip == 0:
        try:
            # Convert frame to RGB (DeepFace works best with RGB images)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Analyze the frame with DeepFace
            analysis = DeepFace.analyze(rgb_frame, actions=["age", "gender"], enforce_detection=True)

            detections = []  # Reset previous detections
            if isinstance(analysis, list):
                for face in analysis:
                    age = face.get('age', "Unknown")
                    gender = face.get('dominant_gender', "Unknown")
                    region = face.get('region', {})

                    x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 100), region.get('h', 100)
                    
                    # Ensure bounding box is within frame
                    x, y, w, h = max(0, x), max(0, y), min(frame.shape[1], w), min(frame.shape[0], h)

                    detections.append((x, y, w, h, age, gender))
                    last_detection_time = time.time()

        except Exception as e:
            print("Face detection error:", e)

    # Display last detections for 3 seconds to reduce flickering
    if time.time() - last_detection_time < detection_time:
        for (x, y, w, h, age, gender) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Gender: {gender}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Real-Time Age & Gender Detection", frame)

    # Exit on 'q' key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break

# Release resources properly
cap.release()
cv2.destroyAllWindows()
print("Webcam released. Program exited.")
