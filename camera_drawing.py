import cv2
import numpy as np
import mediapipe as mp

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 607.5)
draw_canvas = None  # To store drawing

prev_x, prev_y = 0, 0
drawing = False  # You can use thumb touching index to toggle if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    if draw_canvas is None:
        draw_canvas = np.zeros_like(frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get index finger tip coordinates
        index_tip = hand_landmarks.landmark[8]
        x, y = int(index_tip.x * w), int(index_tip.y * h)

        # Optional: Show fingertip
        cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)

        if drawing:
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y
            cv2.line(draw_canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)
            prev_x, prev_y = x, y
        else:
            prev_x, prev_y = 0, 0

    # Merge drawing with webcam frame
    frame = cv2.addWeighted(frame, 0.5, draw_canvas, 0.5, 0)

    # Instructions
    cv2.putText(frame, "Press 'd' to toggle drawing | Press 'c' to clear | Press 'ESC' to exit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

    cv2.imshow("Air Drawing", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('d'):
        drawing = not drawing
    elif key == ord('c'):
        draw_canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()
