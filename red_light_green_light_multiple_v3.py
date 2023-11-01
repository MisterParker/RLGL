import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Constants
MINOR_THRESHOLD = 10  # Change these values based on your sensitivity requirements
MAJOR_THRESHOLD = 30
POSE_CONNECTIONS = [...]

# ... (Other functions remain the same)

cap = cv2.VideoCapture(1)  # Use default camera
frame_count = 0

# Output video settings
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 5.0, (640, 480))  # Adjusted to 5 FPS and 640x480 resolution

previous_keypoints_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lower the resolution
    frame = cv2.resize(frame, (640, 480))  # Adjust as needed
    
    # Process every 5th frame for 5 FPS
    frame_count += 1
    if frame_count % 5 != 0:
        continue

    all_current_keypoints = detect_keypoints_in_regions(frame)

    # ... (Rest of the processing code remains the same)

    # Write the processed frame to the output video
    out.write(frame)

    cv2.imshow('Red Light & Green Light', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
