import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp

# Load the trained model for digits
model = load_model("models/asl_digit_model_simple.h5")

# Mediapipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Map the numeric labels to digits (0 to 9)
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Start the video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video. Exiting...")
        break

    # Flip the frame horizontally for natural hand orientation
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box coordinates for the hand
            h, w, c = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Add some padding to the bounding box
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Crop the hand region
            hand_roi = frame[y_min:y_max, x_min:x_max]

            # Preprocess the cropped hand region
            if hand_roi.size > 0:
                hand_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                hand_roi = cv2.resize(hand_roi, (28, 28))  # Resize to match the model input
                hand_roi = hand_roi.astype('float32') / 255  # Normalize pixel values
                hand_roi = np.expand_dims(hand_roi, axis=-1)  # Add channel dimension
                hand_roi = np.expand_dims(hand_roi, axis=0)  # Add batch dimension

                # Predict the sign language digit
                prediction = model.predict(hand_roi)
                predicted_label = np.argmax(prediction)

                # Ensure predicted_label is within the correct range
                if 0 <= predicted_label < len(class_labels):
                    predicted_digit = class_labels[predicted_label]
                else:
                    predicted_digit = "Unknown"

                # Display the prediction on the frame
                cv2.putText(frame, f"Prediction: {predicted_digit}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # Display the frame
    cv2.imshow("Sign Language Digit Recognition", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
