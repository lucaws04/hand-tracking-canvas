import cv2
import numpy as np
import mediapipe as mp

# Setup MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)

eraser_mode = False
eraser_thickness = 30  # Set a default eraser thickness

# Drawing state
prev_x, prev_y = 0, 0
drawing = False
brush_thickness = 5

# Colors
colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
selected_color_index = 0

# Scaling factor to reduce the size of all components
scale_factor = 0.5  # Adjust this factor to make everything smaller (0.7 means 30% smaller)

# Mouse tracking
mouse_click = None
mouse_pos = (0, 0)

def mouse_callback(event, x, y, flags, param):
    global mouse_click, mouse_pos
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click = (x, y)

cv2.namedWindow("Canvas Drawing", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Canvas Drawing", mouse_callback)

# Overlay function
def overlay_image_alpha(background, overlay, x, y):
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > bw or y + h > bh:
        return
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c] +
            (1 - alpha) * background[y:y+h, x:x+w, c]
        )

# Load cursor image
cursor_img = cv2.imread(r"Lucas\images\pointer.png", cv2.IMREAD_UNCHANGED)
if cursor_img is None:
    print("Error: Cursor image not loaded. Check path.")
    exit()

cursor_size = 40
cursor_img = cv2.resize(cursor_img, (int(cursor_size * scale_factor), int(cursor_size * scale_factor)))

# Pinch distance
def get_pinch_distance(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    return np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))

# Canvas placeholder (will resize per frame)
draw_canvas = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get current frame size
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Define desired canvas size (scale for screen size)
    desired_width = 1080  # You can set this to your preferred window width
    desired_height = 608  # Aspect ratio: 16:9

    # Scale frame to fit the canvas size
    scale_x = desired_width / w
    scale_y = desired_height / h
    scale = min(scale_x, scale_y)

    # Resize the frame and recalculate the drawing canvas size
    frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

    if draw_canvas is None or draw_canvas.shape[:2] != frame_resized.shape[:2]:
        draw_canvas = np.ones((frame_resized.shape[0], frame_resized.shape[1], 3), dtype=np.uint8) * 255

    # Set the drawing parameters
    prev_x, prev_y = 0, 0
    cursor_x, cursor_y = -1, -1

    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        index_tip = hand_landmarks.landmark[8]
        x, y = int(index_tip.x * w * scale), int(index_tip.y * h * scale)
        cursor_x, cursor_y = x, y

        if not drawing:
            pinch_distance = get_pinch_distance(hand_landmarks.landmark)
            brush_thickness = int(np.interp(pinch_distance, [0.02, 0.15], [2, 30]))
            cursor_size = int(brush_thickness * 2 * scale_factor)  # Scaling the cursor size too

        if drawing or eraser_mode:
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y
            if eraser_mode:
                cv2.line(draw_canvas, (prev_x, prev_y), (x, y), (255, 255, 255), eraser_thickness)
            else:
                cv2.line(draw_canvas, (prev_x, prev_y), (x, y), colors[selected_color_index], brush_thickness)
            prev_x, prev_y = x, y
        else:
            prev_x, prev_y = 0, 0


   # Blend the canvas and the webcam feed to create a transparent effect
    alpha = 0.4  # Transparency level (0 = fully transparent, 1 = fully opaque)
    display_frame = cv2.addWeighted(frame_resized, alpha, draw_canvas, 1 - alpha, 0)

    # Draw color sidebar with hover
    sidebar_x = int(w * scale) - int(40 * scale_factor)  # Adjust for sidebar
    mx, my = mouse_pos

    palette_box_size = int(40 * scale_factor)  # Scale palette box size

    for i, color in enumerate(colors):
        center_y = 70 + i * (palette_box_size + int(20 * scale_factor))  # Adjust the margin
        center = (sidebar_x, center_y)
        radius = palette_box_size // 2

        is_hovered = (mx - center[0])**2 + (my - center[1])**2 <= (radius + 6)**2
        is_selected = i == selected_color_index

        # Border style
        if is_selected:
            border_color = (0, 0, 0)
            thickness = int(3 * scale_factor)
        elif is_hovered:
            border_color = (100, 100, 100)
            thickness = int(2 * scale_factor)
        else:
            border_color = (200, 200, 200)
            thickness = int(1 * scale_factor)

        cv2.circle(display_frame, center, radius + 4, border_color, thickness)
        cv2.circle(display_frame, center, radius, color, -1)

        if mouse_click:
            if (mx - center[0]) ** 2 + (my - center[1]) ** 2 <= radius ** 2:
                selected_color_index = i
                mouse_click = None

    # Draw cursor
    if cursor_x != -1 and not drawing:
        resized_cursor = cv2.resize(cursor_img, (cursor_size, cursor_size))
        offset_x = cursor_x - cursor_size // 2
        offset_y = cursor_y - cursor_size // 2
        overlay_image_alpha(display_frame, resized_cursor, offset_x, offset_y)

    # Text
    mode_text = "Eraser" if eraser_mode else "Brush"
    cv2.putText(display_frame, f"Mode: {mode_text} | Brush: {brush_thickness}px | 'D' to draw | 'C' to clear | 'E' to erase | W/S or Click to change color | ESC to exit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Display frame
    cv2.imshow("Canvas Drawing", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('d'):
        drawing = not drawing
    elif key == ord('c'):
        draw_canvas[:] = 255
    elif key == ord('w'):
        selected_color_index = (selected_color_index - 1) % len(colors)
    elif key == ord('s'):
        selected_color_index = (selected_color_index + 1) % len(colors)
    elif key == ord('e'):
        eraser_mode = not eraser_mode


cap.release()
cv2.destroyAllWindows()
