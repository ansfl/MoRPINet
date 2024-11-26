from typing import Optional

from torch import nn


class TrainModel:

    def __init__(self, config):

        self.model: Optional[nn.Module] = None
        self.loss: Optional[callable] = None
        self.optimizer: Optional[callable] = None
        self.scheduler: Optional[callable] = None

        # load configuration to neural network model
        self.set_values(**config['d_net'].to_dict())
        # set model to cuda (or cpu if not available)
        self.model.to(config.device)

    def set_values(self, model, optimizer, scheduler, loss, **kwargs):

        if model['args'] is not None:
            self.model = model['type'](**model['args'])
        else:
            self.model = model['type']()
        self.optimizer = optimizer['type'](params=self.model.parameters(), lr=optimizer['lr'])
        self.scheduler = scheduler['type'](optimizer=self.optimizer, **scheduler['args'])
        if isinstance(loss, type):
            self.loss = loss()
        else:
            self.loss = loss
