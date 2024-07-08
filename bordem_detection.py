import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Load dlib's pre-trained face detector
detector = dlib.get_frontal_face_detector()

# Load the pre-trained facial landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# Function to compute the Eye Aspect Ratio (EAR)
def compute_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Function to compute the Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (3.0 * D)
    return mar


# Define thresholds for EAR and MAR
EAR_THRESHOLD = 0.23
MAR_THRESHOLD = 0.31

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])

        # Extract eye landmarks
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # Compute EAR for both eyes
        left_ear = compute_ear(left_eye)
        right_ear = compute_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Debugging: Print EAR value
        print(f"Left EAR: {left_ear}, Right EAR: {right_ear}, Average EAR: {ear}")

        # Extract mouth landmarks
        mouth = landmarks[48:68]
        mar = mouth_aspect_ratio(mouth)

        # Debugging: Print MAR value
        print(f"MAR: {mar}")

        # Check for boredom indicators
        if ear < EAR_THRESHOLD:
            cv2.putText(frame, "Boredom detected: Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                        2)
        if mar > MAR_THRESHOLD:
            cv2.putText(frame, "Boredom detected: Yawning", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw facial landmarks
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
