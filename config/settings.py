import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "mobilenet_sign_model.h5")
SMOOTHER_PATH = os.path.join(MODELS_DIR, "smoother.h5")   # optional LSTM smoother
LABELS_PATH = os.path.join(DATA_DIR, "labels.txt")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Recognition smoothing / voting
SMOOTH_WINDOW = 7
VOTE_THRESHOLD = 4
LETTER_PAUSE_SEC = 0.8

# Smoother settings
SMOOTHER_SEQ_LEN = 12  # length of probability sequences passed to optional smoother

# Minimum detection confidence for MediaPipe
MIN_DETECTION_CONFIDENCE = 0.6

# UI
CAMERA_INTERFACE_WIDTH = 700   # px approximate width for camera display (moderate)
CAMERA_INTERFACE_HEIGHT = 420  # px approximate height