import os

import numpy as np
import pandas as pd
import torch.utils.data
from sklearn.model_selection import train_test_split

from scripts.configs.config import Config
from scripts.data_loader.steps_dataloader import StepDataset
from scripts.data_loader import sync_imu_dir_path, raw_imu_dir_path
from scripts.utils.dataset_loader_utils import load_rtk


class DatasetLoader:
    """
    RTK data is in NED coordinate, therefore, the 0-axis in all the positions arrays is the North and 1-axis is East.
    The angles defined as the angle between the north to the east. psi = arctan(dNorth / dEast) [0,2pi]
    """

    def __init__(self, config: Config):
        self.config = config

        self.train_dataset = None
        self.validation_dataset = None

        self.imu_train = None
        self.target_gt_train = None

        self.imu_val = None
        self.target_gt_val = None

        self.imu_test = None
        self.rtk_test = None
        self.full_angles_gt_test = None
        self.target_gt_test = None
        self.test_task_names = None

        self.traj_idx = None

        self.create_dataset()
        print('dataset loading - Done!')

    def load_data(self, batch_size):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0)
        validation_loader = torch.utils.data.DataLoader(self.validation_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=0)

        return train_loader, validation_loader

    def create_dataset(self):
        self.collect_data()

        self.train_dataset = StepDataset(self.config,
                                         self.imu_train,
                                         self.target_gt_train)

        self.validation_dataset = StepDataset(self.config,
                                              self.imu_val,
                                              self.target_gt_val)

    def collect_data(self):
        max_num_recordings_train = 0
        try:
            max_num_recordings_train += len(os.listdir(sync_imu_dir_path))
        except FileNotFoundError:
            print("Error: missing dataset files")
            exit()

        max_xdots = 5
        max_num_recordings_test = len(self.config.test_missions) * max_xdots
        cnt_test_recordings = 0
        cnt_train_recordings = 0

        self.imu_train = np.zeros(max_num_recordings_train, dtype=object)
        self.target_gt_train = np.zeros(max_num_recordings_train, dtype=object)

        self.imu_test = np.zeros(max_num_recordings_test, dtype=object)
        self.target_gt_test = np.zeros(max_num_recordings_test, dtype=object)
        self.rtk_test = np.zeros(max_num_recordings_test, dtype=object)
        self.full_angles_gt_test = np.zeros(max_num_recordings_test, dtype=object)

        self.test_task_names = np.zeros(max_num_recordings_test, dtype=object)

        # list of indices of the beginning of each rtk-trajectory to be removed for angles training
        rtk_traj_idx_train = [0]

        duration_test = 0
        # test set
        for c in self.config.test_missions:
            rtk_ned = load_rtk(mission_name=c)
            if rtk_ned is None:
                continue

            duration = 0
            imu_c_list = [f for f in os.listdir(sync_imu_dir_path) if f.startswith(f'IMU_{c}_') and f.endswith('.npy')]
            for imu_file in imu_c_list:
                xdot = int(imu_file[-5])

                imu = self.load_imu(mission_name=c,
                                    xdot_num=xdot,
                                    imu_file=imu_file)

                rtk_target, imu_samples, rtk_ang, rtk_ned_cut, duration = self.gen_data(rtk_data=rtk_ned,
                                                                                        imu_data=imu,
                                                                                        overlap_test=self.config.overlap_test)

                self.imu_test[cnt_test_recordings] = imu_samples
                self.target_gt_test[cnt_test_recordings] = rtk_target
                self.rtk_test[cnt_test_recordings] = rtk_ned_cut
                self.full_angles_gt_test[cnt_test_recordings] = rtk_ang

                self.test_task_names[cnt_test_recordings] = f'{c}_{xdot}'

                cnt_test_recordings += 1

            duration_test += duration

        duration_train = 0
        duration_train_total = 0
        # train set
        for c in self.config.train_missions:
            rtk_ned = load_rtk(mission_name=c)
            if rtk_ned is None:
                continue

            duration = 0
            imu_c_list = [f for f in os.listdir(sync_imu_dir_path) if f.startswith(f'IMU_{c}_') and f.endswith('.npy')]

            for imu_file in imu_c_list:
                xdot = int(imu_file[-5])

                imu = self.load_imu(mission_name=c,
                                    xdot_num=xdot,
                                    imu_file=imu_file)

                rtk_target, imu_samples, rtk_ang, _, duration = self.gen_data(rtk_data=rtk_ned,
                                                                              imu_data=imu)

                self.imu_train[cnt_train_recordings] = imu_samples
                self.target_gt_train[cnt_train_recordings] = rtk_target

                # save the index of the first window in each trajectory to delete because the angle used is the IC
                rtk_traj_idx_train.append(len(self.target_gt_train))

                cnt_train_recordings += 1

                duration_train_total += duration

            duration_train += duration

        self.concat_and_cut_data(cnt_train_recordings, cnt_test_recordings)

        self.traj_idx = rtk_traj_idx_train[:-1]

        self.config.data_to_file.add_to_res_file(id_key='test',
                                                 msg=f'test set total duration (one IMU)',
                                                 data=duration_test)

        self.config.data_to_file.add_to_res_file(id_key='test_avg',
                                                 msg=f'test set avg duration',
                                                 data=duration_test/len(self.config.test_missions))

        self.config.data_to_file.add_to_res_file(id_key='train',
                                                 msg=f'train set total duration (one IMU)',
                                                 data=duration_train)

        self.config.data_to_file.add_to_res_file(id_key='train',
                                                 msg=f'train set total duration (all IMU)',
                                                 data=duration_train_total)

    def concat_and_cut_data(self, cnt_train: int, cnt_test: int):
        self.imu_train = np.concatenate(self.imu_train[:cnt_train], axis=0)
        self.target_gt_train = np.concatenate(self.target_gt_train[:cnt_train], axis=0)

        self.clean_data()

        target = self.target_gt_train

        x_train, x_val, y_train, y_val = train_test_split(self.imu_train, target, train_size=0.9)

        self.imu_train = x_train
        self.target_gt_train = y_train

        self.imu_val = x_val
        self.target_gt_val = y_val

        self.imu_test = self.imu_test[:cnt_test]
        self.target_gt_test = self.target_gt_test[:cnt_test]

        self.rtk_test = self.rtk_test[:cnt_test]
        self.test_task_names = self.test_task_names[:cnt_test]
        self.full_angles_gt_test = self.full_angles_gt_test[:cnt_test]

    def load_imu(self, mission_name: str, xdot_num: int = 1, imu_file: str = None):
        try:
            imu = np.load(sync_imu_dir_path / imu_file)
            imu = np.asarray(imu)
        except FileNotFoundError:
            imu = None
            print(f'ERROR: file not found - {imu_file}')

        try:
            imu_raw = pd.read_csv(raw_imu_dir_path / f'raw_imu_{mission_name}_{xdot_num}.csv', skiprows=1, dtype=str)
            imu_raw = pd.DataFrame.to_numpy(imu_raw)[:, 5:-1].astype(float)
            imu_raw = np.delete(imu_raw, np.unique(np.argwhere(np.isnan(imu_raw))[:, 0]), axis=0)
            imu_raw = np.delete(imu_raw, np.unique(np.argwhere(np.isinf(imu_raw))[:, 0]), axis=0)
        except FileNotFoundError:
            imu_raw = None
            print(f'ERROR: file not found - raw_imu_{mission_name}_{xdot_num}.csv')

        if self.config.do_calibration and imu_raw is not None:
            bias = self.config.calibration_func(x=imu_raw,
                                                start_time=3,
                                                duration=3,
                                                frequency=self.config.imu_freq,
                                                f_z_axis_index=2,
                                                g=self.config.g)
            imu = imu - bias

        if self.config.gyro_in_degrees:
            if imu is not None:
                imu[:, 3:] = np.radians(imu[:, 3:])

        return imu

    def gen_data(self,
                 rtk_data: np.ndarray,
                 imu_data: np.ndarray,
                 overlap_test: bool = True):
        rtk_target, imu_samples, rtk_ang, rtk_ned_cut, duration = None, None, None, None, None
        return rtk_target, imu_samples, rtk_ang, rtk_ned_cut, duration

    def clean_data(self):
        pass

