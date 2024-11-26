import os
from pathlib import Path

weights_dir_path = Path(__file__).parent.parent.parent / 'weights'
if not os.path.exists(weights_dir_path):
    os.mkdir(weights_dir_path)
