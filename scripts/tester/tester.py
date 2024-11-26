import os
from typing import Optional

import numpy as np
import torch

from scripts.configs.config import Config
from scripts.data_loader import sync_imu_dir_path
from scripts.data_loader.data_loader_class import DatasetLoader
from scripts.models.ahrs import AHRS
from scripts.models.ins import INS
from scripts.models.morpi import MoRPI
from scripts.tester import weights_dir_path
from scripts.trainer.train_func_class import TrainModel
from scripts.utils.evaluation_metrics import get_metrics, get_error_in_percents
from scripts.utils.graphs import Graphs
from scripts.utils.results_to_file import ResultFile


class Tester:
    def __init__(self,
                 config: Config,
                 dataloader: DatasetLoader,
                 model: TrainModel,
                 graphs: Graphs,
                 result_file: ResultFile):

        self.config = config
        self.graphs = graphs
        self.result_file = result_file

        self.imu = dataloader.imu_test
        self.pos_gt = dataloader.rtk_test
        self.test_missions = dataloader.test_task_names

        self.dataset_gt = dataloader.target_gt_test

        self.model = model

        self.morpi = MoRPI(self.config)
        self.ins = INS(self.config)
        self.ahrs = AHRS(self.config)

        self.recon_eval_metrics = {}
        self.nn_gt = {}
        self.morpi_gt = {'MoRPI-A': {}, 'MoRPI-G': {}}
        self.ins_gt = {'INS-3D': {}, 'INS-2D': {}}
        self.traj_recon = {}

        self.all_pred_dnet = []
        self.all_gt_dnet = []
        self.all_pred_hnet = []
        self.all_gt_hnet = []
        self.net_eval_metrics = {}

        self.results_dict = {}
        """
        the result dictionary fields:
        [task]-> [error type]->[units]->[model]
              -> [duration]
              -> [length]
              -> [net mode]->[RMSE type]
        """
        self.result_avg_mission = {}
        self.results_straight_dict = {}
        self.result_straight_avg_mission = {}

    def run_test(self, **kwargs):
        if self.config.test:
            self.load_weights()

        callback = kwargs.get('callback')
        if callback is None:
            print('Error: run_test is missing callback function')
            exit()

        if type(callback) == list and len(callback) > 1:
            calling_callback = callback[0]
            passing_callback = callback[1:]
            kwargs.pop('callback', None)
            kwargs['callback'] = passing_callback
        elif type(callback) == list and len(callback) == 1:
            calling_callback = callback[0]
        else:
            calling_callback = callback

        calling_callback(**kwargs)

        self.test_ins_straight_trajectory()

        self.evaluate_reconstruct_trajectories()
        self.evaluate_networks_performance()
        self.add_avg_to_dict()
        self.update_graphs()
        self.update_results_file()

    def wrapper_missions(self, **kwargs):
        callback = kwargs.get('callback')
        if callback is None:
            print('Error: wrapper_missions is missing callback function')
            exit()

        for mission in enumerate(self.test_missions):
            self.call_func(callback, mission)

    @staticmethod
    def call_func(callback, mission):
        if type(callback) == list and len(callback) > 1:
            callback[0](callback=callback[1:], mission=mission)
        elif type(callback) == list and len(callback) == 1:
            callback[0](mission=mission)
        else:
            callback(mission=mission)

    def test_trajectory(self, **kwargs):
        mission = kwargs.get('mission')
        if mission is None or type(mission) not in [str, tuple]:
            print('Error: test_trajectory is missing mission or wrong format')
            exit()

        if type(mission) is str:
            try:
                mission_number = np.where(self.test_missions == mission)[0].item()
            except ValueError:
                print('Error: Wrong mission name to test')
                exit()
            mission = (mission_number, mission)
        # add trajectory to dictionary
        self.add_task_to_dict(mission, self.results_dict)

        # add GT trajectory values to dictionary
        if mission[1] not in self.nn_gt.keys():
            self.nn_gt[mission[1]] = self.pos_gt[mission[0]][::self.config.window_size, :2]

        self.network_reconstruct(mission)

        self.morpi_reconstruct(mission)

        self.ins_reconstruct(self.results_dict, mission)

    def test_ins_straight_trajectory(self):
        """
        run INS on the straight trajectories
        :return:
        """
        if self.config.INS:
            for c in self.config.INS_missions:
                if c is not None:
                    imu_c_list = [f for f in os.listdir(sync_imu_dir_path) if
                                  f.startswith(f'IMU_{c}_') and f.endswith('.npy')]
                    if not imu_c_list:
                        continue

                    total_duration = 0
                    for idx, imu_file in enumerate(imu_c_list):
                        xdot = int(imu_file[-5])
                        task_name = f'{c}_{xdot}'

                        task = tuple([idx, task_name])
                        self.add_task_to_dict(task, self.results_straight_dict)

                        total_duration += self.ins_reconstruct(self.results_straight_dict, task)

                    self.config.data_to_file.add_to_res_file(id_key='ins',
                                                             msg=f'INS straight set total duration (all IMU)',
                                                             data=total_duration)
                else:
                    print('Error: missing mission to test INS')
                    exit()

    def add_task_to_dict(self, task, res_dict: dict):
        if task[1] not in res_dict.keys():
            res_dict[task[1]] = {'mean error': {'meters': {}}, self.config.net_mode: {}}
        else:
            res_dict[task[1]][self.config.net_mode] = {}

    def network_reconstruct(self, task: tuple):
        if self.config.test:

            if self.config.net_mode not in self.traj_recon.keys():
                self.traj_recon[self.config.net_mode] = {}

            with torch.no_grad():
                imu = torch.tensor(self.imu[task[0]].astype(float)).to(self.config.device, dtype=torch.float)

                output = self.get_values_from_network(imu=imu, task=task)

            # dead-reckoning for each step
            pos, pos_ref, psi = self.get_network_trajectory(nn_output=output, task=task)

            self.get_network_statistic(reconstruct_pos=pos, reconstruct_yaw=psi, nn_output=output, task=task)

            if task[1] not in self.nn_gt.keys():
                self.nn_gt[task[1]] = pos_ref
            self.traj_recon[self.config.net_mode][task[1]] = pos

    def morpi_reconstruct(self, task: tuple):
        if self.config.MoRPI:
            if not all(morpi_type in self.traj_recon for morpi_type in ['MoRPI-A', 'MoRPI-G']):
                self.traj_recon.update({'MoRPI-A': {}, 'MoRPI-G': {}})
            # skip if we already ran MoRPI on this task
            if 'MoRPI-A' in self.results_dict[task[1]]['mean error']['meters'].keys():
                return

            for mode in ['sf', 'gyro']:
                pos_morpi, \
                 gt_morpi, \
                 mean_morpi_err = self.morpi.run_morpi(task=task,
                                                       data_type=mode,
                                                       graph=(False, False, False))

                morpi_type = 'MoRPI-A' if mode == 'sf' else 'MoRPI-G'
                self.results_dict[task[1]]['mean error']['meters'][f'{morpi_type}'] = mean_morpi_err

                self.morpi_gt[morpi_type][task[1]] = gt_morpi
                self.traj_recon[morpi_type][task[1]] = pos_morpi

    def ins_reconstruct(self, results_dict, task):
        if self.config.INS:
            if not all(ins_type in self.traj_recon for ins_type in ['INS-3D', 'INS-2D']):
                self.traj_recon.update({'INS-3D': {}, 'INS-2D': {}})
            # skip if we already ran INS on this task
            if 'INS-3D' in results_dict[task[1]]['mean error']['meters'].keys():
                return

            for mode in [True, False]:
                pos_ins, \
                 gt_ins, \
                 mean_ins_err, \
                 duration = self.ins.run_ins(task=task,
                                             use_three_dim=mode,
                                             graph=(False, False, False))

                ins_type = 'INS-3D' if mode else 'INS-2D'
                results_dict[task[1]]['mean error']['meters'][f'{ins_type}'] = mean_ins_err

                self.ins_gt[ins_type][task[1]] = gt_ins
                self.traj_recon[ins_type][task[1]] = pos_ins

            return duration

    def load_weights(self):
        self.model.model.eval()

        # load weights if not trained in this run
        if not self.config.train:
            # load d-net weights if needed
            if self.config.net_mode is not None:
                try:
                    self.model.model.load_state_dict(torch.load(weights_dir_path / self.config.net_mode /
                                                                self.config.dnet_weights_file,
                                                                map_location=torch.device(self.config.device)))
                    print(f'D-net weights were loaded')
                    print(f'weights file: {weights_dir_path / self.config.net_mode / self.config.dnet_weights_file}')
                except RuntimeError:
                    print('D-Net weights are random, Error was occurred')
        else:
            print('using weights for D-Net from training')

    def get_network_trajectory(self, nn_output: np.ndarray, task: tuple):
        """
        0-axis is the North axis and 1-axis is the East axis
        :param nn_output: array with the step distances from the NN and the heading from AHRS
        :param task: the name and number of the current trajectory to test
        :return:
        """
        pos = np.zeros((len(self.dataset_gt[task[0]]) + 1, 2))
        pos_ref = np.zeros_like(pos)

        pos, pos_ref, psi = self.get_trajectory(task, pos, pos_ref, nn_output)

        return pos, pos_ref, psi

    def evaluate_reconstruct_trajectories(self):
        # network results
        if self.config.test:

            recon_arr = {}
            for net_mode in self.traj_recon.keys():

                if 'MoRPI-' in net_mode:
                    gt_arr = np.array([x for x in self.morpi_gt[net_mode].values()], dtype=object)
                elif 'INS' in net_mode:
                    gt_arr = np.array([x for x in self.ins_gt[net_mode].values()], dtype=object)
                else:
                    gt_arr = np.array([x[:-1, :] for x in self.nn_gt.values()], dtype=object)

                recon_arr[net_mode] = np.array([x for x in self.traj_recon[net_mode].values()], dtype=object)
                self.recon_eval_metrics[net_mode] = get_metrics(gt_arr, recon_arr[net_mode])

    def evaluate_networks_performance(self):
        if self.all_pred_dnet:
            all_pred_dnet_concat = np.concatenate(np.array(self.all_pred_dnet, dtype=object), axis=0)
            all_gt_dnet_concat = np.concatenate(np.array(self.all_gt_dnet, dtype=object), axis=0)
            self.net_eval_metrics['Dnet'] = get_metrics(all_gt_dnet_concat,
                                                        all_pred_dnet_concat)

            steps_error_percents = get_error_in_percents(all_gt_dnet_concat, self.net_eval_metrics['Dnet'])
            self.net_eval_metrics['Dnet percents'] = steps_error_percents

    def add_avg_to_dict(self):
        self.results_dict['avg'] = {'mean error': {'meters': {}}, self.config.net_mode: {}}
        self.results_straight_dict['avg'] = {'mean error': {'meters': {}}, self.config.net_mode: {}}

        net_mode_options = self.results_dict[list(self.results_dict.keys())[0]]['mean error']['meters'].keys()
        self.result_avg_mission = {'mean error': {key: {} for key in net_mode_options}}

        net_mode_options = self.results_straight_dict[list(self.results_straight_dict.keys())[0]]['mean error']['meters'].keys()
        self.result_straight_avg_mission = {'mean error': {key: {} for key in net_mode_options}}

        error = 'mean error'
        for net_mode in self.results_dict[list(self.results_dict.keys())[0]][error]['meters'].keys():
            try:
                avg_error = np.mean(
                    [self.results_dict[task][error]['meters'][net_mode]
                     for task in list(self.results_dict.keys())[:-1]])

                self.results_dict['avg'][error]['meters'][net_mode] = avg_error
            except KeyError:
                pass

            for mission in self.config.test_missions:
                self.result_avg_mission[error][net_mode][mission] = np.mean(
                                                                    [self.results_dict[task][error]['meters'][net_mode]
                                                                     for task in list(self.results_dict.keys())[:-1]
                                                                     if task.startswith(mission)])

            self.result_avg_mission[error][net_mode]['avg'] = self.results_dict['avg'][error]['meters'][net_mode]

        for net_mode in self.results_straight_dict[list(self.results_straight_dict.keys())[0]][error]['meters'].keys():
            try:
                avg_error = np.mean(
                    [self.results_straight_dict[task][error]['meters'][net_mode]
                     for task in list(self.results_straight_dict.keys())[:-1]])

                self.results_straight_dict['avg'][error]['meters'][net_mode] = avg_error
            except KeyError:
                pass

            for mission in self.config.INS_missions:
                self.result_straight_avg_mission[error][net_mode][mission] = np.mean(
                                                                    [self.results_straight_dict[task][error]['meters'][net_mode]
                                                                     for task in list(self.results_straight_dict.keys())[:-1]
                                                                     if task.startswith(mission)])

            self.result_straight_avg_mission[error][net_mode]['avg'] = self.results_straight_dict['avg'][error]['meters'][net_mode]

    def update_graphs(self):
        self.graphs.recon_traj = self.traj_recon
        self.graphs.gt_traj = self.nn_gt

    def update_results_file(self):
        self.result_file.recon_eval_metrics = self.recon_eval_metrics
        self.result_file.net_eval_metrics = self.net_eval_metrics

        self.result_file.res_avg_dict = self.result_avg_mission
        self.result_file.res_straight_dict = self.results_straight_dict
        self.result_file.res_straight_avg_dict = self.result_straight_avg_mission

    # functions to overwrite:
    def get_values_from_network(self, imu: np.ndarray, task: tuple) -> np.ndarray:
        output = np.array([])
        return output

    def get_network_statistic(self,
                              reconstruct_pos: np.ndarray,
                              reconstruct_yaw: Optional[np.ndarray],
                              nn_output: np.ndarray,
                              task: tuple):
        pass

    def get_trajectory(self, task, pos, pos_ref, nn_output) -> tuple:
        pass
