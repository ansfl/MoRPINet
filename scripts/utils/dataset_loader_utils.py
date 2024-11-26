import numpy as np

from scripts.data_loader import sync_rtk_dir_path


def load_rtk(mission_name: str):
    try:
        rtk = np.load(sync_rtk_dir_path / f'ned_{mission_name}.npy')
        rtk = np.asarray(rtk)
    except FileNotFoundError:
        rtk = None
        print(f'ERROR: file not found - ned_{mission_name}.npy')

    return rtk
