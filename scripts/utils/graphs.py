from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np

from scripts.configs.config import Config
from scripts.utils.dataset_loader_utils import load_rtk


class Graphs:
    def __init__(self, config: Config):
        self.config = config

        self.recon_traj = None
        self.gt_traj = None

        self.show_plot_train_loss: bool = True
        self.show_plot_rtk_trajectories: bool = self.config.plot_missions
        self.show_plot_reconstruct_trajectory: bool = self.config.plot_reconstruct_missions

    def plot_train_loss(self, model: str, train_loss: list, val_loss: list):
        if self.show_plot_train_loss:
            model_config = 'd_net'

            plt.figure()
            plt.plot(np.arange(self.config[model_config].epochs), train_loss, label='train loss')
            plt.plot(np.arange(self.config[model_config].epochs), val_loss, label='validation loss')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.grid(True)
            plt.legend()

    def plot_rtk_trajectories(self, mission: Union[str, list],
                              show_graph: bool = False,
                              idx_list: Optional[np.array] = None,
                              step_size: int = 1):
        """
        :param mission: str or list. can get the values:
                'all' - plot all the trajectories used for training and testing the model
                'test' - plot all the trajectories used to test the model
                'train' - plot all the trajectories used to train the model
                'ins' - plot all the trajectories used to test the INS model
                or name of specific mission to plot
                or list of specific missions to plot
        :param show_graph: show the graph despite config
        :param idx_list: list of idx to add text next to them
        :param step_size: size of step to skip samples
        """
        if self.show_plot_rtk_trajectories or show_graph:
            mission_list = self.get_mission_list(mission)

            for n, c in enumerate(mission_list):
                traj = load_rtk(c)
                plt.figure()
                plt.plot(traj[::step_size, 1], traj[::step_size, 0], label=f'GT train trajectory {n}')
                if idx_list is not None:
                    for idx in idx_list:
                        plt.annotate(idx,
                                     (traj[idx*step_size, 1], traj[idx*step_size, 0]),
                                     textcoords='offset points',
                                     xytext=(0, 10),
                                     ha='center')
                plt.xlabel('East [m]')
                plt.ylabel('North [m]')
                plt.axis('equal')
                plt.grid(True)
                plt.legend()

    def plot_reconstruct_trajectory(self, net_mode: Union[str, list]):

        if self.show_plot_reconstruct_trajectory:

            if net_mode == 'all':
                mode_list = self.recon_traj.keys()
            elif type(net_mode) == str:
                mode_list = [net_mode]
            else:
                mode_list = net_mode

            test_missions = self.get_mission_list('test')
            model_missions = self.recon_traj[list(mode_list)[0]].keys()
            mission_list = [m for m in model_missions if m.split('_')[0] in test_missions]

            num_mission = np.arange(3, len(mission_list)+3)
            for n, m in enumerate(mission_list):
                plt.figure()
                plt.plot(self.gt_traj[m][:, 1], self.gt_traj[m][:, 0], 'k', label=f'GT for {num_mission[n]}')
                plt.plot(self.gt_traj[m][[0, -1], 1], self.gt_traj[m][[0, -1], 0], 'ok')
                for model in mode_list:
                    plt.plot(self.recon_traj[model][m][:, 1], self.recon_traj[model][m][:, 0], label=f'{model} for {num_mission[n]}')
                plt.xlabel('East [m]')
                plt.ylabel('North [m]')
                plt.axis('equal')
                plt.grid(True)
                plt.legend()

    def get_mission_list(self, mission: Union[str, list]):
        if mission == 'all':
            mission_list = self.config.train_missions + self.config.test_missions
        elif mission == 'test':
            mission_list = self.config.test_missions
        elif mission == 'ins':
            mission_list = self.config.INS_missions
        elif mission == 'train':
            mission_list = self.config.train_missions
        else:
            if type(mission) == list:
                mission_list = mission
            else:
                mission_list = [mission]

        return mission_list

    def show(self, mission: Union[str, list] = 'all', net_mode: Union[str, list] = 'all'):

        self.plot_rtk_trajectories(mission)

        self.plot_reconstruct_trajectory(net_mode)

        plt.show()


