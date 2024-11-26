from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

from scripts.configs.config import Config
from scripts.data_loader import sync_imu_dir_path, raw_imu_dir_path
from scripts.data_loader.data_loader_class import DatasetLoader
from scripts.tester.tester import Tester
from scripts.trainer.train_func_class import TrainModel
from scripts.utils.graphs import Graphs
from scripts.utils.results_to_file import ResultFile
from scripts.utils.test_utils import downsampling_pos, dead_reckoning


class MoRPINetTester(Tester):
    def __init__(self, config: Config,
                 dataloader: DatasetLoader,
                 model: TrainModel,
                 graphs: Graphs,
                 result_file: ResultFile):

        super().__init__(config,
                         dataloader,
                         model,
                         graphs,
                         result_file)

    def get_values_from_network(self, imu: np.ndarray, task: tuple):
        # get distance values
        if self.config.net_mode is not None:
            pred_dists = torch.squeeze(self.model.model(imu)).cpu().numpy()
            self.all_pred_dnet.append(pred_dists)
            self.all_gt_dnet.append(self.dataset_gt[task[0]][:, 0])
        else:  # GT
            pred_dists = self.dataset_gt[task[0]][:, 0]

        # get angles values
        if self.config.net_mode is not None:  # AHRS
            pred_ang = self.get_ahrs_ang(task=task)[1:].T
        else:  # GT
            pred_ang = self.dataset_gt[task[0]][:, 1]

        output = np.hstack([pred_dists.reshape(-1, 1), pred_ang.reshape(-1, 1)])

        return output

    def get_network_statistic(self,
                              reconstruct_pos: np.ndarray,
                              reconstruct_yaw: Optional[np.ndarray],
                              nn_output: np.ndarray,
                              task: tuple):

        pred_dists = nn_output[:, 0]

        # average step size and step error
        avg_step_gt = np.sqrt(np.mean(self.dataset_gt[task[0]][:, 0] ** 2))
        rmse_steps = mean_squared_error(self.dataset_gt[task[0]][:, 0], pred_dists, squared=False)
        steps_error_percents = rmse_steps / avg_step_gt * 100

        self.results_dict[task[1]][self.config.net_mode]['RMSE steps'] = rmse_steps
        self.results_dict[task[1]][self.config.net_mode]['RMSE steps percents'] = steps_error_percents

        duration = len(self.dataset_gt[task[0]]) / (self.config.window_size * 10)
        length = np.sum(self.dataset_gt[task[0]][:, 0])

        # get every second RTK sample corresponding to the window size
        gt_intrp = downsampling_pos(self.pos_gt[task[0]], self.config.window_size)
        # get the position error
        pos_err = np.sqrt(np.sum((reconstruct_pos[:, :2] - gt_intrp[:, :2]) ** 2, axis=1))
        mean_pos_err = np.sqrt(np.mean(pos_err ** 2))
        self.results_dict[task[1]]['mean error']['meters'][self.config.net_mode] = mean_pos_err

        self.results_dict[task[1]]['duration'] = duration
        self.results_dict[task[1]]['length'] = length

    def get_ahrs_ang(self, task: tuple):
        try:
            imu_df = np.load(sync_imu_dir_path / f'IMU_{task[1]}.npy')
        except FileNotFoundError:
            print("ERROR: couldn't load IMU file")
            exit()

        try:
            raw_df = pd.read_csv(raw_imu_dir_path / f'raw_imu_{task[1]}.csv', skiprows=1)
            raw_df = pd.DataFrame.to_numpy(raw_df)[:, 5:-1].astype(float)
        except FileNotFoundError:
            print('RAW IMU file not found')
            raw_df = None

        samples = np.arange(0, len(imu_df), self.config.window_size * self.config.imu_rtk_ratio)

        init_yaw = self.dataset_gt[task[0]][0, 1]

        ang = self.ahrs.calculate_heading(imu=imu_df[:, :6],
                                          raw_imu=raw_df,
                                          initial_angles=(0, 0, init_yaw),
                                          samples=samples)

        return ang

    def get_trajectory(self, task, pos, pos_ref, nn_output):
        pred_dist = nn_output[:, 0]
        pred_ang = nn_output[:, 1].astype(float)

        psi = np.zeros(pos.shape[0] - 1)
        # initial heading
        psi[0] = self.dataset_gt[task[0]][0, 1]
        psi_ref = np.zeros_like(psi)
        psi_ref[0] = psi[0]

        for i in range(len(self.dataset_gt[task[0]])):
            pos[i + 1, :] = dead_reckoning(pos[i], pred_dist[i], psi[i])
            pos_ref[i + 1, :] = dead_reckoning(pos_ref[i], self.dataset_gt[task[0]][i, 0], psi_ref[i])
            if i == len(self.dataset_gt[task[0]]) - 1:
                break

            psi[i + 1] = pred_ang[i + 1]
            psi_ref[i + 1] = psi_ref[i] + self.dataset_gt[task[0]][i + 1, 1]

        return pos, pos_ref, psi
