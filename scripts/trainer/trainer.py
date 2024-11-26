from typing import Optional

import numpy as np
import os
import torch

from scripts.configs.config import Config
from scripts.data_loader.data_loader_class import DatasetLoader
from scripts.trainer import weights_dir_path
from scripts.trainer.train_func_class import TrainModel
from scripts.utils.graphs import Graphs


class Trainer:
    def __init__(self,
                 config: Config,
                 dataloader: DatasetLoader,
                 timestamp: Optional[str],
                 graphs: Graphs,
                 model: TrainModel):

        self.config = config
        self.dataloader = dataloader
        self.train_loader = None
        self.validation_loader = None

        self.graphs = graphs

        self.timestamp = timestamp

        self.model = model

        if not os.path.exists(weights_dir_path / self.config.net_mode):
            os.mkdir(weights_dir_path / self.config.net_mode)

    def train_loop(self, model_name: str):

        train_loss_list, validation_loss_list = [], []

        for epoch in range(self.config[model_name].epochs):
            train_loss = []

            # Train loop
            self.model.model.train()
            for i, data in enumerate(self.train_loader):
                imu, target = data
                try:
                    target = target[:, 0]
                    target = torch.unsqueeze(target, 1)
                except IndexError:
                    target = torch.unsqueeze(target, 1)
                imu = imu.to(self.config.device, dtype=torch.float)
                target = target.to(self.config.device, dtype=torch.float)

                prediction = self.model.model(imu)
                loss = self.model.loss(prediction, target)

                train_loss.append(loss.item())

                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()

            if self.config[model_name].scheduler is not None:
                self.model.scheduler.step(np.mean(np.array(train_loss)))

            train_loss_list.append(np.mean(np.array(train_loss)))

            # Validation loop
            self.model.model.eval()
            with torch.no_grad():
                val_loss = []

                for data in self.validation_loader:
                    imu, target = data
                    imu = imu.to(self.config.device, dtype=torch.float)
                    target = target.to(self.config.device, dtype=torch.float)
                    try:
                        target = target[:, 0]
                    except IndexError:
                        target = torch.unsqueeze(target, 1)

                    output = self.model.model(imu)
                    loss = self.model.loss(output, target)

                    val_loss.append(loss.item())

            validation_loss_list.append(np.mean(np.array(val_loss)))

            # print progress
            if (epoch + 1) % 5 == 0:
                print(f'epoch: [{epoch + 1}/{self.config[model_name].epochs}], '
                      f'train_loss: {np.mean(train_loss):.4f} val_loss: {np.mean(val_loss):.4f}')

        return train_loss_list, validation_loss_list

    def train(self):

        if self.config.train:
            print('D-Net: start of training...')
            self.train_loader, self.validation_loader = self.dataloader.load_data(self.config.d_net.batch_size)
            d_train_loss_list, d_val_loss_list = self.train_loop(model_name='d_net')

            last_dnet_weights = weights_dir_path / self.config.net_mode / f'Dnet_weights_{self.timestamp}.pth'
            torch.save(self.model.model.state_dict(), last_dnet_weights)

            self.graphs.plot_train_loss(model='d_net',
                                        train_loss=d_train_loss_list,
                                        val_loss=d_val_loss_list)
