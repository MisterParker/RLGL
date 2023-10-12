import cv2
import numpy as np
import mediapipe as mp
import serial 
import time 

# Initialize mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Constants
MINOR_THRESHOLD = 10  # Change these values based on your sensitivity requirements
MAJOR_THRESHOLD = 30

# Define connections between keypoints (based on mediapipe's documentation)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), 
    (5, 6), (6, 8), (9, 10), (11, 12), (12, 14), 
    (14, 16), (11, 13), (13, 15), (15, 17), (0, 9), 
    (0, 11), (9, 11), (2, 9), (3, 10), (2, 12), (3, 13)
]

arduino = serial.Serial(port='/dev/cu.usbmodem1101', baudrate = 115200, timeout=0.1)

def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    # time.sleep(0.05)
    data = arduino.readline()
    return data

def detect_keypoints(frame):
    # Convert the BGR image to RGB before processing
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    keypoints = []
    if results.pose_landmarks:
        keypoints = [[landmark.x * frame.shape[1], landmark.y * frame.shape[0]] for landmark in results.pose_landmarks.landmark]
        return np.array(keypoints)
    return []

def detect_keypoints_in_regions(frame, num_regions=4):
    """Detect keypoints in vertical regions of the frame."""
    height, width, _ = frame.shape
    region_width = width // num_regions

    all_keypoints = []
    for i in range(num_regions):
        region = frame[:, i*region_width:(i+1)*region_width]
        region_keypoints = detect_keypoints(region)
        
        # Offset keypoints to global frame coordinates
        for k in region_keypoints:
            k[0] += i * region_width

        all_keypoints.append(region_keypoints)

    return all_keypoints

def calculate_movement(prev_keypoints, curr_keypoints):
    if len(prev_keypoints) == 0 or len(curr_keypoints) == 0:
        return 0
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

# Add this function to calculate the center of keypoints
def get_center_of_keypoints(keypoints):
    if len(keypoints) == 0:
        return None
    mean_x = np.mean([kp[0] for kp in keypoints])
    mean_y = np.mean([kp[1] for kp in keypoints])
    return (int(mean_x), int(mean_y))


def calculate_angle_from_center(center, frame_width, fov=75):
    # Calculate the horizontal offset from the center of the frame
    offset = center[0] - frame_width / 2
    # Calculate angle
    angle = (offset / frame_width) * fov
    return angle

cap = cv2.VideoCapture(1)  # Use default camera
# cap = cv2.VideoCapture('test_video.mp4')

# Output video settings
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

previous_keypoints_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    all_current_keypoints = detect_keypoints_in_regions(frame)

    for idx, current_keypoints in enumerate(all_current_keypoints):
        if idx < len(previous_keypoints_list) and len(previous_keypoints_list[idx]) > 0:
            movement = calculate_movement(previous_keypoints_list[idx], current_keypoints)
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

            # Calculate the center of keypoints
            center = get_center_of_keypoints(current_keypoints)
            if center:  # If center exists
                # Mark the center on the image
                cv2.putText(frame, f"Person {idx + 1}", (center[0]-60, center[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, center, 5, (0, 0, 0), -1)  # Black dot for center
                cv2.putText(frame, f"{center}", (center[0]-50, center[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # # Print the center to the terminal
                print(f"Center: {center}")

                angle = calculate_angle_from_center(center, frame.shape[1])
                new_angle = -1*int(angle) + 90
                # new_angle = 125
                print(write_read(str(new_angle)))
                print(f"angle sent to arduino : {new_angle}")
                cv2.putText(frame, f"{angle}", (center[0]-30, center[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                print(f"Angle for person {idx + 1}: {angle:.2f} degrees")

            # Display movement text at the bottom center
            cv2.putText(frame, movement_text, (frame.shape[1]//2 - 50, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the title "PlaySmart" at the top center
        cv2.putText(frame, "PlaySmart", (frame.shape[1]//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)



    previous_keypoints_list = all_current_keypoints

    # Write the processed frame to the output video
    out.write(frame)

    cv2.imshow('Red Light & Green Light', frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('p'):
        # Print the centers to the terminal when 'p' key is pressed
        for current_keypoints in all_current_keypoints:
            center = get_center_of_keypoints(current_keypoints)
            if center:
                print(f"Center: {center}")

cap.release()
out.release()
cv2.destroyAllWindows()