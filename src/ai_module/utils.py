from typing import Any
from torch import cuda, save, load

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