import cv2
import numpy as np
import mediapipe as mp
import time

# Setup MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Zoom control
prev_pinch_distance = None
zoom_level = 1.0
target_zoom = 1.0
smoothing_factor = 0.1

# Locking behavior
zoom_locked = False
lock_time = 0
lock_duration = 1.0  # seconds
pinch_active = False

# Detect pinch gesture
def get_pinch_distance(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    return np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))

def apply_zoom_effect(frame, zoom):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    new_w = int(w / zoom)
    new_h = int(h / zoom)
    x1 = max(center[0] - new_w // 2, 0)
    y1 = max(center[1] - new_h // 2, 0)
    x2 = min(center[0] + new_w // 2, w)
    y2 = min(center[1] + new_h // 2, h)
    cropped = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h))
    return zoomed

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 607.5)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    current_time = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            curr_distance = get_pinch_distance(hand_landmarks.landmark)

            if prev_pinch_distance is None:
                prev_pinch_distance = curr_distance

            distance_diff = curr_distance - prev_pinch_distance

            # Tweak thresholds separately
            pinch_in_threshold = 0.015
            pinch_out_threshold = 0.010  # more sensitive to small decreases
            min_pinch_gap = 0.03  # avoid jitter when fingers too close

            pinch_active = False

            if curr_distance > min_pinch_gap:
                distance_diff = curr_distance - prev_pinch_distance

                if distance_diff > pinch_in_threshold:
                    target_zoom = min(2.5, target_zoom + 0.05)
                    cv2.putText(frame, "Zooming In", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 255, 0), 4, cv2.LINE_AA)
                    pinch_active = True

                elif distance_diff < -pinch_out_threshold:
                    target_zoom = max(0.5, target_zoom - 0.05)
                    cv2.putText(frame, "Zooming Out", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 255, 255), 4, cv2.LINE_AA)
                    pinch_active = True

            if pinch_active:
                zoom_locked = False
                lock_time = current_time
                prev_pinch_distance = curr_distance
            else:
                if not zoom_locked and (current_time - lock_time > lock_duration):
                    zoom_locked = True
                    prev_pinch_distance = None

            if not zoom_locked:
                prev_pinch_distance = curr_distance

    # Smooth zooming
    zoom_level = (1 - smoothing_factor) * zoom_level + smoothing_factor * target_zoom

    # Apply visual zoom effect
    frame = apply_zoom_effect(frame, zoom_level)

    cv2.imshow('Gesture Demo', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
