from pathlib import Path
from config import HYPERPARAMETERS

import os
import datetime



def file_exists(filepath: str) -> bool:
    return os.path.isfile(filepath)


def directory_exists(file_dir: str) -> bool:
    return os.path.isdir(file_dir)


def join_path(src: str, dst: str) -> str:
    return os.path.join(src, dst)


def list_files(file_dir: str):
    path = Path(file_dir)

    return [str(file) for file in path.rglob("*") if file.is_file()]


def get_log_path(log_dir: str, name: str) -> str:
    base_dir = join_path(log_dir, "runs")
    time_str = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    # Create the params string
    params = "-".join(f"{key}={value}" for key, value in HYPERPARAMETERS.items())

    run_id = f"{name}{params}({time_str})"

    return join_path(base_dir, run_id)
