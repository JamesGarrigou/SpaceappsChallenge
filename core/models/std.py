from argparse import Namespace
from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn as nn

from ..training import cli_train
from ..training.datamodule import RegressionDataModule
from ..training.network import STD
from ..training.optimization_procedure import optim
from ..training.routine import SingleRegression


# fmt: on
class STD(SingleRegression):
    learning_rate: float = None
    tau: float = None
    train_size: int = None

    def __init__(self, config: Union[Dict, Namespace], **kwargs) -> None:
        if isinstance(config, Namespace):
            config = vars(config)

        n_features: int = config.get("n_features")
        hidden_units: int = config.get("hidden_units")

        super().__init__()
        self.save_hyperparameters(config)

        self.model = STD(n_features, 2 * hidden_units)

        # to log the graph
        self.example_input_array = torch.randn(1, n_features)

    def configure_optimizers(self) -> dict:
        return optim(self, self.learning_rate, self.train_size)

    @property
    def criterion(self) -> nn.Module:
        return nn.GaussianNLLLoss()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model.forward(input)


if __name__ == "__main__":
    root = Path(__file__).parent.absolute()
    cli_train(STD, RegressionDataModule, root, "std")
