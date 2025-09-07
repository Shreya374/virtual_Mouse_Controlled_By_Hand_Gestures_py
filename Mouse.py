import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

# Screen size
screen_w, screen_h = pyautogui.size()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Smoothing parameters
prev_x, prev_y = 0, 0
smoothening = 7

mouse_active = True  # Toggle mouse on/off

def get_finger_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror image
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            # Index finger tip
            x1 = int(lm[8].x * w)
            y1 = int(lm[8].y * h)

            # Middle finger tip
            x2 = int(lm[12].x * w)
            y2 = int(lm[12].y * h)

            # Thumb tip
            x_thumb = int(lm[4].x * w)
            y_thumb = int(lm[4].y * h)

            # Move mouse
            if mouse_active:
                screen_x = np.interp(x1, (100, w - 100), (0, screen_w))
                screen_y = np.interp(y1, (100, h - 100), (0, screen_h))
                curr_x = prev_x + (screen_x - prev_x) / smoothening
                curr_y = prev_y + (screen_y - prev_y) / smoothening
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

            # Draw index finger
            cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

            # Left click gesture: index + thumb close
            if get_finger_distance((x1, y1), (x_thumb, y_thumb)) < 40:
                pyautogui.click()
                time.sleep(0.3)

            # Right click gesture: index + middle close
            if get_finger_distance((x1, y1), (x2, y2)) < 40:
                pyautogui.click(button='right')
                time.sleep(0.3)

            # Scroll gesture: index + middle far apart (up/down)
            if y2 - y1 > 50:
                pyautogui.scroll(-30)
                time.sleep(0.1)
            elif y1 - y2 > 50:
                pyautogui.scroll(30)
                time.sleep(0.1)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, "Virtual Mouse (Press 'm' to toggle)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if mouse_active:
        cv2.putText(frame, "Mouse: ON", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Mouse: OFF", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Virtual Mouse", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('m'):
        mouse_active = not mouse_active  # Toggle mouse on/off

cap.release()
cv2.destroyAllWindows()