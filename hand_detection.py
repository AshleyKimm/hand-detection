import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
webcam = cv2.VideoCapture(0)

hands = mp_hands.Hands()
canvas = None
prev_x, prev_y = 0, 0
while webcam.isOpened():
    success, image = webcam.read()
    results = hands.process(image)

    
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    image = cv2.flip(image, 1)

    if canvas is None:
        canvas = np.zeros_like(image)
    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, mark the image as not writeable to pass by reference
    image.flags.writeable = False

    # Process the image and detect hands
    results = hands.process(image)

    # Draw the hand annotations on the image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    draw = False
    # Check if any hands were detected and draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, connections = mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_x = int(index_tip.x * image.shape[1])
            index_y = int(index_tip.y * image.shape[0])
            thumb_x = int(thumb_tip.x * image.shape[1])
            thumb_y = int(thumb_tip.y * image.shape[0])

            middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            middle_x = int(middle_pip.x * image.shape[1])
            middle_y = int(middle_pip.y * image.shape[0])


            cv2.line(image, (index_x, index_y), (thumb_x, thumb_y), (0, 255, 0), 2)
            cv2.putText(image, 'Index Finger', (index_x, index_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, 'Thumb', (thumb_x, thumb_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(image, (index_x, index_y), 5, (0, 0, 255), -1)

            distance = np.sqrt((index_x - middle_x) ** 2 + (index_y - middle_y) ** 2)
            cv2.putText(image, f'Distance: {distance}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if distance < 0.05:
                draw = not draw
                prev_x, prev_y = 0, 0
                cv2.putText(image, 'toggle drawing', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if draw:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = index_x, index_y

                cv2.line(image, (prev_x, prev_y), (index_x, index_y), (255, 0, 0), 2)
                prev_x, prev_y = index_x, index_y

            if not draw:
                prev_x, prev_y = 0, 0

                
    else:
        cv2.putText(image, 'No hands detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)
    



    # Display the resulting frame.
    image = cv2.flip(image, 1)
    cv2.imshow('Hand Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break
