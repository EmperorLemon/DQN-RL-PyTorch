from pathlib import Path
import os


def file_exists(filepath: str) -> bool:
    return os.path.isfile(filepath)


def directory_exists(file_dir: str) -> bool:
    return os.path.isdir(file_dir)


def join_path(src: str, dst: str) -> str:
    return os.path.join(src, dst)


def list_files(file_dir: str):
    path = Path(file_dir)

    return [str(file) for file in path.rglob("*") if file.is_file()]
