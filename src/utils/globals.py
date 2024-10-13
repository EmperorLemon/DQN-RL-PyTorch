import os

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

BASE_DIR = os.path.abspath(os.path.dirname(os.getcwd()))
ASSETS_DIR = os.path.abspath(os.path.join(BASE_DIR, "assets"))
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "logs"))
MODEL_DIR = os.path.abspath(os.path.join(ASSETS_DIR, "models"))
CHECKPOINT_DIR = os.path.abspath(os.path.join(MODEL_DIR, "checkpoints"))
DATA_DIR = os.path.abspath(os.path.join(ASSETS_DIR, "data"))

# Ensure necessary directories exist
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
