import yaml
import torch

from scripts.configs import yml_cnfg_file_path, yml_dnet_cnfg_file_path
from scripts.configs.train_config import TrainConfig
import scripts.utils.utils as utils


class Config:
    # running options
    train: bool
    test: bool
    MoRPI: bool
    INS: bool

    # missions lists
    train_missions: list
    test_missions: list
    MoRPI_train_missions: list
    INS_missions: list

    # graphs
    plot_missions: bool
    plot_reconstruct_missions: bool

    # dataset options
    window_size: int
    overlap: int
    overlap_test: bool

    imu_freq: int
    rtk_freq: int
    imu_rtk_ratio: int
    g: float

    do_calibration: bool
    do_ins_calibration: bool
    calibration_func: callable
    do_lpf: bool

    gyro_in_degrees: bool
    madgwick_avg_angle: bool

    device: str
    net_mode: str
    d_net: TrainConfig
    dnet_weights_file: str
    morpi_turn_threshold: int

    def __init__(self):
        Config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(Config.device)

        self.data_to_file = utils.DataToFile()

        cls = type(self)  # Reference the class
        with open(yml_cnfg_file_path, 'r') as file:
            config_data = yaml.safe_load(file)  # Parse YAML file
        for key, value in config_data.items():
            setattr(cls, key, value)

        cls.imu_rtk_ratio = int(cls.imu_freq / cls.rtk_freq)

        try:
            cls.calibration_func = getattr(utils, config_data['calibration_func'])
        except AttributeError:
            print("ERROR: couldn't find the calibration function")
            exit()

        cls.d_net = TrainConfig(config_file=yml_dnet_cnfg_file_path)

    @classmethod
    def __getitem__(cls, item):
        return cls.__dict__[item]

    def __repr__(self):
        return f"Config({type(self).__dict__})"

    def to_dict(self) -> dict:
        d = {key: value for key, value in Config.__dict__.items() if not key.startswith('__') and not callable(value)}
        d_self = {key: value for key, value in self.__dict__.items() if
                  not key.startswith('__') and not callable(value)}
        d.update(d_self)
        return d
