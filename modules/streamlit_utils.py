import cv2
import time
import os
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque, Counter

from modules.utils import make_square_roi, draw_hand_landmarks
from config.settings import (
    MODEL_PATH, LABELS_PATH, IMG_SIZE,
    SMOOTH_WINDOW, VOTE_THRESHOLD, LETTER_PAUSE_SEC
)
from modules.tts_engine import speak_text


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cv2.setUseOptimized(True)


class SignLanguageRecognizer:
    def __init__(self):

        # Safety checks 
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("❌ Model missing. Train the model first.")

        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError("❌ Labels file missing.")

        # Load model + labels
        self.model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH) as f:
            self.labels = [l.strip() for l in f.readlines()]

        # Mediapipe (Fast mode)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,    
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        # Buffers
        self.buffer = deque(maxlen=SMOOTH_WINDOW)
        self.text = ""

        # Control letter repetition + speech
        self.last_letter_time = time.time()
        self.last_spoken_letter = None


    # FRAME PROCESSING
    def process_frame(self, frame):

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if not res.multi_hand_landmarks:
            return frame, None, 0.0

        lm = res.multi_hand_landmarks[0]

        # Extract square ROI
        xs = [int(p.x * w) for p in lm.landmark]
        ys = [int(p.y * h) for p in lm.landmark]

        sx, sy, ex, ey = make_square_roi(min(xs), max(xs), min(ys), max(ys), w, h)
        roi = frame[sy:ey, sx:ex]

        # Draw bounding box + landmarks 
        frame = draw_hand_landmarks(frame, lm, mp_hands, mp_draw, (sx, sy, ex, ey))

        if roi.size == 0:
            return frame, None, 0.0

        # CLASSIFY ROI
        img = cv2.resize(roi, IMG_SIZE).astype("float32") / 255.0
        preds = self.model.predict(np.expand_dims(img, 0), verbose=0)

        idx = np.argmax(preds[0])
        confidence = float(preds[0][idx])

        # Reject unstable detections
        if confidence < 0.35:
            return frame, None, confidence

        label = self.labels[idx]

        # Smoothing + speak (once)
        self.update_text(label, lm)

        # Draw typed text
        cv2.putText(
            frame, self.text, (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.05,
            (255, 255, 0), 2
        )

        return frame, label, confidence


    # UPDATE TEXT (SMOOTH, NO REPEAT SPEAKING)
    def update_text(self, label, lm):

        self.buffer.append(label)

        # Not enough history yet
        if len(self.buffer) < self.buffer.maxlen:
            return

        most, freq = Counter(self.buffer).most_common(1)[0]

        if freq < VOTE_THRESHOLD:
            return

        now = time.time()

        # Wait between letters (debounce)
        if now - self.last_letter_time < LETTER_PAUSE_SEC:
            return

        # SPACE SIGN (OPEN PALM)
        if self.is_open_palm(lm):
            if self.last_spoken_letter != "SPACE":
                self.text += " "
                self.last_spoken_letter = "SPACE"
                self.last_letter_time = now
            return

        # Speak ONLY if new letter
        if most != self.last_spoken_letter:
            self.text += most
            speak_text(most)     
            self.last_spoken_letter = most

        self.last_letter_time = now


    # OPEN PALM DETECTION → SPACE
    def is_open_palm(self, lm):
        tip_ids = [8, 12, 16, 20]
        extended = sum(
            lm.landmark[t].y < lm.landmark[t - 2].y for t in tip_ids
        )
        return extended >= 3


    # API FUNCTIONS
    def get_text(self):
        return self.text

    def clear(self):
        self.text = ""
        self.buffer.clear()
        self.last_spoken_letter = None