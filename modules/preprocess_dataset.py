# Preprocess data

import os
import cv2
import mediapipe as mp
from modules.utils import make_square_roi, ensure_dirs
from config.settings import RAW_DIR, PROCESSED_DIR, IMG_SIZE, MIN_DETECTION_CONFIDENCE

mp_hands = mp.solutions.hands

def rebuild_processed():
    ensure_dirs(RAW_DIR, PROCESSED_DIR)
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE
    )

    for label in os.listdir(RAW_DIR):
        raw_label = os.path.join(RAW_DIR, label)
        proc_label = os.path.join(PROCESSED_DIR, label)
        os.makedirs(proc_label, exist_ok=True)

        files = sorted([f for f in os.listdir(raw_label) if f.lower().endswith((".jpg", ".png"))])
        saved = 0
        for fname in files:
            path = os.path.join(raw_label, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if not res.multi_hand_landmarks:
                continue
            lm = res.multi_hand_landmarks[0]
            xs = [int(p.x * w) for p in lm.landmark]
            ys = [int(p.y * h) for p in lm.landmark]
            sx, sy, ex, ey = make_square_roi(min(xs), max(xs), min(ys), max(ys), w, h)
            sx, sy, ex, ey = map(int, (sx, sy, ex, ey))
            roi = img[sy:ey, sx:ex]
            if roi.size == 0:
                continue
            roi_resized = cv2.resize(roi, IMG_SIZE)
            out_name = os.path.join(proc_label, f"{label}_{saved}.jpg")
            cv2.imwrite(out_name, roi_resized)
            saved += 1
        print(f"[{label}] processed {saved} images.")

    hands.close()

if __name__ == "__main__":
    rebuild_processed()