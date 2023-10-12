import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Constants
MINOR_THRESHOLD = 10  # Change these values based on your sensitivity requirements
MAJOR_THRESHOLD = 25

previous_keypoints = {}

# Define connections between keypoints (based on mediapipe's documentation)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), 
    (5, 6), (6, 8), (9, 10), (11, 12), (12, 14), 
    (14, 16), (11, 13), (13, 15), (15, 17), (0, 9), 
    (0, 11), (9, 11), (2, 9), (3, 10), (2, 12), (3, 13)
]

def detect_keypoints(frame):
    # Convert the BGR image to RGB before processing
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    all_keypoints = []
    if results.pose_landmarks:
        for pose_landmarks in results.pose_landmarks:
            keypoints = [[landmark.x * frame.shape[1], landmark.y * frame.shape[0]] for landmark in pose_landmarks.landmark]
            all_keypoints.append(np.array(keypoints))
    return all_keypoints

def calculate_movement(prev_keypoints, curr_keypoints):
    return np.linalg.norm(prev_keypoints - curr_keypoints, axis=1).max()

def draw_skeleton(frame, keypoints, color):
    for point in keypoints:
        cv2.circle(frame, tuple(map(int, point)), 5, color, -1)
    
    # Draw lines connecting keypoints
    for connection in POSE_CONNECTIONS:
        start, end = connection
        if start < len(keypoints) and end < len(keypoints):
            cv2.line(frame, tuple(map(int, keypoints[start])), tuple(map(int, keypoints[end])), color, 2)
    return frame

cap = cv2.VideoCapture(1)  # Use default camera

# Output video settings
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    all_current_keypoints = detect_keypoints(frame)
    
    for idx, current_keypoints in enumerate(all_current_keypoints):
        if idx < len(previous_keypoints):
            movement = calculate_movement(previous_keypoints[idx], current_keypoints)
            if movement > MAJOR_THRESHOLD:
                color = (0, 0, 255)  # RED
                movement_text = "Red"
            elif movement > MINOR_THRESHOLD:
                color = (0, 165, 255)  # ORANGE
                movement_text = "Orange"
            else:
                color = (0, 255, 0)  # GREEN
                movement_text = "Green"
            draw_skeleton(frame, current_keypoints, color)

        # Display movement text at the bottom center
            cv2.putText(frame, movement_text, (frame.shape[1]//2 - 50, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)

        # Display the title "PlaySmart" at the top center
        cv2.putText(frame, "PlaySmart", (frame.shape[1]//2 - 100, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4, cv2.LINE_AA)

    previous_keypoints = all_current_keypoints
    out.write(frame)
    cv2.imshow('Red Light & Green Light', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release() 
cv2.destroyAllWindows()
