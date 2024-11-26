import numpy as np
import torch.utils.data

from scripts.configs.config import Config
from scripts.data_loader.data_loader_class import DatasetLoader


class DatasetLoaderMoRPINet(DatasetLoader):
    """
    RTK data is in NED coordinate, therefore, the 0-axis in all the positions arrays is the North and 1-axis is East.
    The angles defined as the angle between the north to the east. psi = arctan(dNorth / dEast) [0,2pi]
    """

    def __init__(self, config: Config):
        super().__init__(config)

    def gen_data(self,
                 rtk_data: np.ndarray,
                 imu_data: np.ndarray,
                 overlap_test: bool = True):
        """
        gets position measurements from GPS/RTK and specific force and angular velocity measurements for IMU.
        computes Euclidean distance from two position measurements and match the IMU measurements measured at the
        same time interval
        :param rtk_data: position measurements in time with respect to NED frame [m] (Nx3)
        :param imu_data: IMU (specific force [m/sec^2] and angular velocity[rad/sec]) measurements  (Mx6)
        :param overlap_test: use overlap for either test missions and training mission
        :return: rtk_dists:  (Kx1)
                 rtk_ang: (Kx1)
                 imu_samples: (KxLx6)
        RTK readings:
        North:   |  |   |   |   |   |   |   |   |   |   |   |   |
        East :   |  |   |   |   |   |   |   |   |   |   |   |   |
        Down :   |  |   |   |   |   |   |   |   |   |   |   |   |
             mov_idx<-------------windows_size--------------->sample_size
        """
        if overlap_test:  # test usually do not need overlap
            step = self.config.window_size - self.config.overlap
        else:
            step = self.config.window_size
        n_rtk = len(rtk_data)
        imu_samples = np.zeros((n_rtk, 6, int(self.config.window_size * self.config.imu_rtk_ratio)))
        rtk_dists = np.zeros((n_rtk, 1))
        rtk_ang = np.zeros((n_rtk, 1))
        mov_idx = 0
        dat_idx = 0

        imu_data = imu_data[:, :6]

        while ((mov_idx + self.config.window_size < n_rtk) and
               ((mov_idx + self.config.window_size) * self.config.imu_rtk_ratio < imu_data.shape[0])):

            sample_size = mov_idx + self.config.window_size
            if sample_size * self.config.imu_rtk_ratio - mov_idx * self.config.imu_rtk_ratio < 24:
                break
            rtk_delta = rtk_data[sample_size, :] - rtk_data[mov_idx, :]

            rtk_dists[dat_idx, 0] = np.linalg.norm([rtk_delta[0], rtk_delta[1]])

            yaw = np.round(np.arctan2(rtk_delta[0], rtk_delta[1]), 4)
            if yaw < 0:  # change yaw to range [0, 2pi]
                yaw = yaw + 2 * np.pi
            rtk_ang[dat_idx, 0] = yaw

            imu_samples[dat_idx, :, :] = np.transpose(imu_data[mov_idx * self.config.imu_rtk_ratio:
                                                               sample_size * self.config.imu_rtk_ratio, :])

            mov_idx += step
            dat_idx += 1

        rtk_dists = rtk_dists[0:dat_idx - 1, :]
        rtk_ang = rtk_ang[0:dat_idx - 1, :]
        rtk_d_ang = self.get_delta_ang(rtk_ang)

        imu_samples = imu_samples[0:dat_idx - 1, :, :]
        rtk_data_cut = rtk_data[:mov_idx + self.config.window_size]

        duration = len(rtk_data_cut) / self.config.rtk_freq

        target_gt = np.hstack((rtk_dists, rtk_d_ang))

        return target_gt, imu_samples, rtk_ang, rtk_data_cut, duration

    @staticmethod
    def get_delta_ang(real_ang: np.ndarray):
        """
        the function convert the real angles of each step to differential of successive angles
        the first angle is the initial condition for the angles
        :param real_ang: the direction of each step in the North-East coordinate system
        :return: delta_ang: differential angles
        """
        delta_ang = np.zeros(real_ang.shape, dtype=real_ang.dtype)
        initial_ang = real_ang[0]
        for i in range(1, real_ang.shape[0]):
            d = np.zeros(3)
            d[0] = (real_ang[i] - real_ang[i - 1])
            d[1] = (real_ang[i] + 2 * torch.pi - real_ang[i - 1])
            d[2] = (real_ang[i] - real_ang[i - 1] - 2 * torch.pi)
            d_abs = np.abs(d)  # distance from zero
            d_idx = np.argmin(d_abs)

            delta_ang[i, :] = d[d_idx]
        delta_ang[0] = initial_ang
        return delta_ang

    def clean_data(self):
        # delete first window of every trajectory because the angle used as IC
        if self.traj_idx is not None:
            self.target_gt_train = np.delete(self.target_gt_train, self.traj_idx, axis=0)
            self.imu_train = np.delete(self.imu_train, self.traj_idx, axis=0)

        # delete noisy data
        ang_th = 0.5
        del_idx = np.where(np.abs(self.target_gt_train[:, 1]) > ang_th)[0]
        neighbours_idx = np.array([])
        for idx in del_idx:
            if idx == 0 or idx == self.target_gt_train.shape[0]-1:
                continue
            neighbours_idx = np.append(neighbours_idx, [idx-1, idx+1])
        del_idx = np.unique(np.insert(del_idx, 0, neighbours_idx))
        del_idx = np.unique(np.insert(del_idx, 0, np.where(np.abs(self.target_gt_train[:, 0]) > 0.5)[0]))
        self.target_gt_train = np.delete(self.target_gt_train, del_idx, axis=0)
        self.imu_train = np.delete(self.imu_train, del_idx, axis=0)

