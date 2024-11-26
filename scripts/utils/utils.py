import time
from pathlib import Path
from typing import Optional, Union

import numpy as np


def get_time_str():
    t = time.localtime()
    time_str = f'{str(t.tm_year)}_{str(t.tm_mon)}_{str(t.tm_mday)}_{str(t.tm_hour)}.{str(t.tm_min)}.{str(t.tm_sec)}'
    return time_str


def do_zero_order_calib(self, x: np.ndarray,
                        start_time: float = 0,
                        duration: float = 3,
                        frequency: float = 100,
                        f_z_axis_index: Optional[int] = None,
                        g: float = 9.796):

    start_sample = start_time * frequency
    stop_sample = start_sample + duration * frequency
    t = np.arange(start_sample, stop_sample)

    avg_x = np.mean(x[t], axis=0)
    if f_z_axis_index is not None:
        avg_x[f_z_axis_index] += g

    return avg_x


def load_data(imu_path: Union[Path, str], gt_path: Union[Path, str, None]):
    imu_df = np.load(imu_path)
    if gt_path is None:
        gt_df = None
        length = None
    else:
        gt_df = np.load(gt_path)
        gt_len = np.sqrt(np.sum(np.diff(gt_df, axis=0)**2, axis=1))
        length = np.sum(gt_len)

    return imu_df, gt_df, length


def moving_average(x, w):
    filtered_x = np.zeros_like(x)
    if len(x.shape) == 2:
        for i in range(x.shape[-1]):
            filtered_x[:, i] = np.convolve(x[:, i], np.ones(w), 'same') / w
    else:
        filtered_x = np.convolve(x, np.ones(w), 'same') / w
    return filtered_x


class DataToFile:
    def __init__(self):
        self.data_to_file_list = {}

    def add_to_res_file(self, id_key, msg, data):
        self.data_to_file_list[id_key] = [msg, data]

