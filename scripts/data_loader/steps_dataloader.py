import numpy as np
from torch.utils.data import Dataset

from scripts.configs.config import Config


class StepDataset(Dataset):
    def __init__(self,
                 config: Config,
                 imu_samples: np.ndarray,
                 rtk_gt: np.ndarray):

        self.config = config
        self.imu_samples = imu_samples
        self.rtk_gt = rtk_gt

    def __getitem__(self, idx):
        imu_sample = self.imu_samples[idx]
        rtk_data = self.rtk_gt[idx]

        return imu_sample, rtk_data

    def __len__(self):
        return len(self.rtk_gt)
