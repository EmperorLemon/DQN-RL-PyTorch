from typing import Any
from torch import cuda, save, load

from utils.utils import join_path, list_files

import os

def check_cuda() -> None:
    if cuda.is_available():
        print("CUDA is available")
        print(f"CUDA device count: {cuda.device_count()}")
        print(f"Current CUDA device: {cuda.current_device()}")
        print(f"CUDA device name: {cuda.get_device_name(0)}\n")
    else:
        print("CUDA is not available, using CPU\n")
        
        
def save_checkpoint(checkpoint: dict, path: str) -> None:
    """Save model state dict and training info"""
    save(checkpoint, path)
    
def load_checkpoint(path: str) -> Any:
    """Load saved model"""
    return load(path, map_location="cpu")

def load_best_checkpoint(checkpoints_dir: str) -> dict:
    best_score = float("-inf")
    best_checkpoint = None
    
    for filepath in list_files(checkpoints_dir):
        cp = load_checkpoint(filepath)
        cp_best_score = cp["best_score"]
        
        if cp_best_score > best_score:
            best_score = cp_best_score
            best_checkpoint = cp
            
            print(f"Highest Score in checkpoint: {os.path.basename(filepath)}")
            print(f"Best Score: {cp_best_score:.2f}")
    
    return best_checkpoint