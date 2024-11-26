from pathlib import Path
from typing import Optional, Union

import torch
import yaml
import torch.nn as nn

import scripts.utils.loss as module_loss
import scripts.models.d_net as d_net_models


class TrainConfig:

    def __init__(self, config_file):
        self.epochs: Optional[int] = None
        self.batch_size: Optional[int] = None

        self.loss: Optional[callable] = None
        self.optimizer: Optional[dict] = None
        self.scheduler: Optional[dict] = None

        self.model: Optional[dict] = None

        self.import_config_from_yaml(config_file)

    def import_config_from_yaml(self, config_file: Union[Path, str]):
        with open(config_file, 'r') as f:
            yml_dict = yaml.safe_load(f)

        self.epochs = yml_dict['epochs']
        self.batch_size = yml_dict['batch_size']

        try:
            self.model = {'type': getattr(d_net_models, yml_dict['model']['type']),
                          'args': yml_dict['model']['args']
                          }
        except AttributeError:
            print('ERROR: Model is not found')
            self.model = None
            exit()

        except KeyError:
            self.model = None
            print('ERROR: model is not defined')
            exit()

        try:
            self.loss = getattr(nn, yml_dict['loss'])
        except AttributeError:
            try:
                self.loss = getattr(module_loss, yml_dict['loss'])
            except AttributeError:
                print("ERROR: couldn't find the Loss function")
                exit()

        except KeyError:
            print('ERROR: Loss function is not defined')
            exit()

        try:
            self.optimizer = {'type': getattr(torch.optim, yml_dict['optimizer']['type']),
                              'lr': yml_dict['optimizer']['args']['lr']
                              }
        except AttributeError:
            print("ERROR: couldn't find the Optimizer function")
            exit()

        except KeyError:
            print('ERROR: Optimizer function is not defined')
            exit()

        try:
            self.scheduler = {'type': getattr(torch.optim.lr_scheduler, yml_dict['scheduler']['type']),
                              'args': yml_dict['scheduler']['args']
                              }
        except AttributeError:
            print('Warning: Scheduler function is not defined')
            self.scheduler = None

        except KeyError:
            self.scheduler = None

    def to_dict(self) -> dict:
        d = {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(value)}
        d['loss'] = self.loss
        return d
