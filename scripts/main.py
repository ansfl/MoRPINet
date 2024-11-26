import matplotlib
import numpy as np
import random
import torch

from scripts.configs.config import Config
from scripts.data_loader.data_loader_morpinet import DatasetLoaderMoRPINet as DL_MoRPINet
from scripts.tester.morpinet_tester import MoRPINetTester
from scripts.trainer.train_func_class import TrainModel
from scripts.trainer.trainer import Trainer
from scripts.utils.graphs import Graphs
from scripts.utils.results_to_file import ResultFile
from scripts.utils.utils import get_time_str

matplotlib.use('TkAgg')


def main():
    config = Config()
    graphs = Graphs(config)

    timestamp = get_time_str()

    model = TrainModel(config)

    data_loader = DL_MoRPINet(config)

    trainer = Trainer(config=config,
                      dataloader=data_loader,
                      model=model,
                      graphs=graphs,
                      timestamp=timestamp)

    result_file = ResultFile(config=config,
                             timestamp=timestamp,
                             model=model)

    tester = MoRPINetTester(config=config,
                            dataloader=data_loader,
                            model=model,
                            graphs=graphs,
                            result_file=result_file)

    trainer.train()

    tester.run_test(callback=[tester.wrapper_missions, tester.test_trajectory])

    result_file.write_to_file(result_dict=tester.results_dict, show_file=True)

    graphs.show()


if __name__ == '__main__':
    seed = 210

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)

    # # Ensure deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()
