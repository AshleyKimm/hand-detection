import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


webcam = cv2.VideoCapture(0)

hands = mp_hands.Hands()

while webcam.isOpened():
    success, image = webcam.read()

    results = hands.process(image)
    
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, mark the image as not writeable to pass by reference
    image.flags.writeable = False

    # Process the image and detect hands
    results = hands.process(image)

    # Draw the hand annotations on the image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Check if any hands were detected and draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, connections = mp_hands.HAND_CONNECTIONS)

    # Display the resulting frame.
    cv2.imshow('Hand Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break
