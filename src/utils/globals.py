import os

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTION_SET = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)
ACTION_NAMES = {ACTION_UP: "up", ACTION_DOWN: "down", ACTION_LEFT: "left", ACTION_RIGHT: "right"}

BASE_DIR = os.path.abspath(os.path.dirname(os.getcwd()))
ASSETS_DIR = os.path.abspath(os.path.join(BASE_DIR, "assets"))
LOGS_DIR = os.path.abspath(os.path.join(BASE_DIR, "logs"))
MODELS_DIR = os.path.abspath(os.path.join(ASSETS_DIR, "models"))
CHECKPOINTS_DIR = os.path.abspath(os.path.join(MODELS_DIR, "checkpoints"))
DATA_DIR = os.path.abspath(os.path.join(ASSETS_DIR, "data"))

# Ensure necessary directories exist
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
