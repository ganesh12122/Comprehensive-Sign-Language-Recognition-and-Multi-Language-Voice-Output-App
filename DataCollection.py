import cv2
import numpy as np
import os
import mediapipe as mp

# Extract Keypoints
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils  # Import mediapipe drawing utilities


# Function to draw landmarks and bounding box
def draw_landmarks_and_bbox(frame, results, left_bbox, right_bbox):
    # Draw face landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # Draw left hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Draw right hand landmarks
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Draw bounding box for left hand region
    if left_bbox is not None:
        cv2.rectangle(frame, (int(left_bbox[0]), int(left_bbox[1])), (int(
            left_bbox[0] + left_bbox[2]), int(left_bbox[1] + left_bbox[3])), (255, 0, 0), 2)
    # Draw bounding box for right hand region
    if right_bbox is not None:
        cv2.rectangle(frame, (int(right_bbox[0]), int(right_bbox[1])), (int(
            right_bbox[0] + right_bbox[2]), int(right_bbox[1] + right_bbox[3])), (0, 255, 0), 2)


# Function to extract keypoints and bounding box coordinates
def extract_keypoints_and_bbox(results, image_width, image_height):
    keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
    ) if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21 * 3)

    left_hand_landmarks = results.left_hand_landmarks
    right_hand_landmarks = results.right_hand_landmarks

    left_bbox = np.zeros(4)
    right_bbox = np.zeros(4)

    if left_hand_landmarks:
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        for landmark in left_hand_landmarks.landmark:
            x, y = landmark.x * image_width, landmark.y * image_height
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        left_bbox = np.array([min_x, min_y, bbox_width, bbox_height])

    if right_hand_landmarks:
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        for landmark in right_hand_landmarks.landmark:
            x, y = landmark.x * image_width, landmark.y * image_height
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        right_bbox = np.array([min_x, min_y, bbox_width, bbox_height])

    return np.concatenate([keypoints, face, lh, rh]), left_bbox, right_bbox


# Path for exported data, numpy arrays
DATA_PATH = os.path.join(os.getcwd(), 'Data')

actions = np.array(['None'])

for action in actions:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
    sequence_number = 121
    sequence_keypoints_and_bbox = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Make detections
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image_height, image_width, _ = frame.shape
        # Extract keypoints and bounding box coordinates
        keypoints, left_bbox, right_bbox = extract_keypoints_and_bbox(
            results, image_width, image_height)
        sequence_keypoints_and_bbox.append(keypoints)
        # Draw landmarks and bounding box on the frame
        draw_landmarks_and_bbox(frame, results, left_bbox, right_bbox)
        cv2.putText(frame, f"Action: {action}, Sequence: {sequence_number}", (
            10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Data Collection', frame)
        if len(sequence_keypoints_and_bbox) == 30:  # Save sequence after collecting 30 frames
            print(
                f"Collecting data for action: {action}, sequence: {sequence_number}")
            np.save(os.path.join(DATA_PATH, actions[0], f"sequence_{sequence_number}.npy"), np.array(
                sequence_keypoints_and_bbox))
            sequence_number += 1
            sequence_keypoints_and_bbox = []  # Reset for the next sequence
            if sequence_number == 151:
                break
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
