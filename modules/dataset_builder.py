# Dataset builder file

import cv2
import os
import time
import mediapipe as mp

from config.settings import RAW_DIR, PROCESSED_DIR, IMG_SIZE, MIN_DETECTION_CONFIDENCE
from modules.utils import (
    make_square_roi,
    draw_hand_landmarks,
    ensure_dirs,
    delete_folder
)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def Capture_images(label, total_images):

    # Ensure directories exist
    ensure_dirs(RAW_DIR, PROCESSED_DIR)

    raw_label_dir = os.path.join(RAW_DIR, label)
    proc_label_dir = os.path.join(PROCESSED_DIR, label)

    os.makedirs(raw_label_dir, exist_ok=True)
    os.makedirs(proc_label_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(" Cannot open webcam. Close other apps using camera.")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0, 
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=0.5
    )

    saved = 0
    frame_count = 0
    prev_t = time.time()

    try:
        while cap.isOpened() and saved < int(total_images):

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            # SAVE RAW FRAME ALWAYS
            raw_path = os.path.join(raw_label_dir, f"{label}_raw_{frame_count}.jpg")
            cv2.imwrite(raw_path, frame)
            frame_count += 1

            # DETECT HAND → ROI CROP
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]

                xs = [int(p.x * w) for p in lm.landmark]
                ys = [int(p.y * h) for p in lm.landmark]

                sx, sy, ex, ey = make_square_roi(min(xs), max(xs), min(ys), max(ys), w, h)
                sx, sy, ex, ey = map(int, (
                    max(0, sx), max(0, sy), min(w, ex), min(h, ey)
                ))

                roi = frame[sy:ey, sx:ex]

                if roi.size > 0:
                    resized = cv2.resize(roi, IMG_SIZE)
                    proc_path = os.path.join(proc_label_dir, f"{label}_{saved}.jpg")
                    cv2.imwrite(proc_path, resized)
                    saved += 1

                    # Draw box + landmarks
                    frame = draw_hand_landmarks(frame, lm, mp_hands, mp_draw, (sx, sy, ex, ey))

            # FPS DISPLAY
            now = time.time()
            fps = 1.0 / max(1e-6, now - prev_t)
            prev_t = now

            cv2.putText(frame, f"Raw: {frame_count}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            cv2.putText(frame, f"Processed: {saved}/{total_images}", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Capture (press 'q' to stop)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

        # AUTO DELETE RAW FOLDER AFTER Processing
        try:
            delete_folder(raw_label_dir)
            print(f"[CLEANUP] Deleted RAW folder: {raw_label_dir}")
        except Exception as e:
            print(f"[WARN] Unable to delete RAW folder: {e}")

    return saved