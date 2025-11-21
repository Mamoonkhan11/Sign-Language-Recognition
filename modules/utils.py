# Helper Functions 

import os
import cv2
import numpy as np
import shutil

def make_square_roi(x_min, x_max, y_min, y_max, w, h, pad=20):
    x_min = max(0, x_min - pad)
    x_max = min(w, x_max + pad)
    y_min = max(0, y_min - pad)
    y_max = min(h, y_max + pad)

    roi_w = x_max - x_min
    roi_h = y_max - y_min
    side = max(1, max(roi_w, roi_h))

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    sx = int(max(0, cx - side // 2))
    ex = int(min(w, cx + side // 2))
    sy = int(max(0, cy - side // 2))
    ey = int(min(h, cy + side // 2))

    return sx, sy, ex, ey


def draw_hand_landmarks(frame, lm, mp_hands, mp_draw, bbox=None):
    if bbox is not None:
        sx, sy, ex, ey = bbox
        cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 200, 0), 2)
    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
    return frame


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_raw_and_processed(frame, label, count, raw_dir, processed_dir, img_size):

    # raw filename
    raw_label_dir = os.path.join(raw_dir, label)
    proc_label_dir = os.path.join(processed_dir, label)
    ensure_dirs(raw_label_dir, proc_label_dir)

    raw_fname = os.path.join(raw_label_dir, f"{label}_raw_{count}.jpg")
    cv2.imwrite(raw_fname, frame)

    return raw_fname

def delete_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        return True
    return False
