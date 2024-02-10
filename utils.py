import os

def safe_open_w(path: str) -> object:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def create_path(path : str) -> None:
    os.makedirs(path, exist_ok=True)
    assert os.path.isdir(path), f'No such dir: {path}'