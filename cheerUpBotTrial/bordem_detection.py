import threading
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time
from concurrent.futures import ThreadPoolExecutor
from newChat import main as startConversation

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
BLINK_THRESHOLD = 40
TIME_WINDOW = 4  # seconds

# Event to determine if voice chat is active
voice_chat_active = threading.Event()

# Initialize blink counter and timer
blink_count = 0
start_time = time.time()

# Thread pool executor for handling voice chat
with ThreadPoolExecutor(max_workers=1) as executor:

    # Capture video from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
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

            # Extract mouth landmarks
            mouth = landmarks[48:68]
            mar = mouth_aspect_ratio(mouth)

            # Check if the person is blinking
            if ear < EAR_THRESHOLD and not voice_chat_active.is_set():
                blink_count += 1
                print(f"Blink count: {blink_count}")

            # Check if blinking threshold is reached within the time window or detect yawning
            elapsed_time = time.time() - start_time
            if elapsed_time > TIME_WINDOW:
                if (blink_count >= BLINK_THRESHOLD or mar > MAR_THRESHOLD) and not voice_chat_active.is_set():
                    print("Boredom detected! Activating voice chat...")
                    voice_chat_active.set()
                    executor.submit(startConversation).add_done_callback(lambda _: voice_chat_active.clear())

                # Reset blink count and timer
                blink_count = 0
                start_time = time.time()

            # Draw facial landmarks (optional)
            for (x, y) in landmarks:
                cv2.circle(small_frame, (x, y), 2, (0, 255, 0), -1)

        # Display the processed frame
        cv2.imshow("Frame", small_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
