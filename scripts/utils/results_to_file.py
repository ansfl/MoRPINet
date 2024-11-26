import os
import webbrowser
from typing import Optional
import pandas as pd

from scripts.configs.config import Config
from scripts.configs.train_config import TrainConfig
from scripts.trainer.train_func_class import TrainModel
from scripts.utils import results_dir_path


class ResultFile:
    def __init__(self, config: Config, timestamp: str, model: TrainModel):
        self.config = config
        self.timestamp = timestamp
        self.model = model

        self.res_file_path = results_dir_path / timestamp.split('.')[0] / f'results_{self.timestamp}.txt'
        self.res_file = None

        self.res_dict: Optional[dict] = None
        self.recon_eval_metrics: Optional[dict] = None
        self.net_eval_metrics: Optional[dict] = None
        self.res_avg_dict: Optional[dict] = None
        self.res_straight_dict: Optional[dict] = None
        self.res_straight_avg_dict: Optional[dict] = None

    def write_to_file(self, result_dict: dict, show_file: bool = False):
        # create folder to save results
        if not os.path.exists(results_dir_path / self.timestamp.split('.')[0]):
            os.mkdir(results_dir_path / self.timestamp.split('.')[0])

        # open file to save results
        self.res_file = open(self.res_file_path, 'w')

        self.res_dict = result_dict

        self.res_file.write('****Results:****\n')
        self.write_tables()

        self.res_file.write('\n****Configs:****\n')
        self.write_configs(self.config.to_dict())

        self.res_file.write('\n****Neural Networks:****\n')
        self.write_nets()

        self.res_file.close()
        if show_file:
            webbrowser.open(str(self.res_file_path))

    def write_tables(self):
        for res_dict, avg_dict in [(self.res_dict, self.res_avg_dict),
                                   (self.res_straight_dict, self.res_straight_avg_dict)]:
            if res_dict is not None:

                error_type = 'mean error'
                units = 'meters'
                df = self.create_table(res_dict, error_type, units)
                self.res_file.write(f'\n{error_type} - {units}\n')
                self.res_file.write(f'\n{df.to_markdown(tablefmt="psql")}\n')

                df = self.create_avg_table(avg_dict, error_type)
                self.res_file.write(f'\n{error_type} - mean values for each trajectory\n')
                self.res_file.write(f'\n{df.to_markdown(tablefmt="psql")}\n')

        self.res_file.write('\n')
        recon_df = pd.DataFrame(self.recon_eval_metrics)
        self.res_file.write(f'reconstructed trajectories evaluation metrics:\n')
        self.res_file.write(f'(How well do the reconstructed trajectories close to the gt)')
        self.res_file.write(f'\n{recon_df.to_markdown(tablefmt="psql")}\n')

        if "MoRPINet" in recon_df.keys() and "MoRPI-A" in recon_df.keys():
            self.res_file.write(f'\nImprovement:')
            self.res_file.write(f'\n{((1 - recon_df["MoRPINet"] / recon_df["MoRPI-A"]) * 100).to_markdown(tablefmt="psql")}\n')

        net_df = pd.DataFrame(self.net_eval_metrics)
        self.res_file.write(f'networks evaluation metrics:\n')
        self.res_file.write(f'(How well do the networks could predict the targets)')
        self.res_file.write(f'\n{net_df.to_markdown(tablefmt="psql")}\n')

    @staticmethod
    def create_table(res_dict, error: str, units: str):
        table_dict = {}
        for task in res_dict.keys():
            table_dict[task] = res_dict[task][error][units]

        df = pd.DataFrame(table_dict)
        return df

    @staticmethod
    def create_avg_table(avg_dict, error_type):
        table_dict = {}
        for net_mode in avg_dict[error_type].keys():
            table_dict[net_mode] = avg_dict[error_type][net_mode]

        df = pd.DataFrame(table_dict)
        return df

    def write_configs(self, config_dict: dict, num_tabs: int = 0):
        for key, value in config_dict.items():
            if type(value) in [dict, TrainConfig]:
                self.res_file.write(f'{key}:\n')
                new_num_tabs = num_tabs + 1
                new_dict = value if type(value) == dict else value.__dict__
                self.write_configs(new_dict, new_num_tabs)
            self.res_file.write('\t'*num_tabs)
            self.res_file.write(f'{key}: {value}\n')

    def write_nets(self):
        if self.model.model is not None:
            self.res_file.write('\nD-Net architecture:\n')
            self.res_file.write(str(self.model.model))
