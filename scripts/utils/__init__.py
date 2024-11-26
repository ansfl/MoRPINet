from pathlib import Path
import os

results_dir_path = Path(__file__).parent.parent.parent / 'results'
if not os.path.exists(results_dir_path):
    os.mkdir(results_dir_path)
