# fmt: off
from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from ..training import cli_train
from ..training.datamodule import RegressionDataModule
from ..training.network import MLP_STD
from ..training.optimization_procedure import optim_regression
from ..training.routine import EnsembleRegression


# fmt: on
class DeepEns(EnsembleRegression):
    learning_rate: float = None
    tau: float = None
    train_size: int = None

    def __init__(self, config: Union[Dict, Namespace]) -> None:
        super().__init__()

        if isinstance(config, Namespace):
            config = vars(config)

        n_features: int = config.get("n_features")
        hidden_units: int = config.get("hidden_units")

        self.n_estimators: int = config.get("n_estimators", 5)

        self.save_hyperparameters(config)

        self.models = nn.ModuleList(
            [
                MLP_STD(n_features, hidden_units=2 * hidden_units)
                for _ in range(self.n_estimators)
            ]
        )

        self.repeat_inputs = False
        self.repeat_targets = False

        # to log the graph
        self.example_input_array = torch.randn(1, n_features)

    def configure_optimizers(self) -> dict:
        if self.learning_rate is None:
            print("Warning ! learning_rate has not been configured")
            self.learning_rate = 1e-3
        if self.train_size is None:
            print("Warning ! train_size has not been configured")
            self.train_size = 1
        return optim_regression(self, self.learning_rate, self.train_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = EnsembleRegression.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("MLP")
        parser.add_argument("--n_estimators", type=int, default=5)
        return parent_parser

    @property
    def criterion(self) -> nn.Module:
        return nn.GaussianNLLLoss()

    def training_step(  # type: ignore
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = batch
        targets = targets.repeat(self.n_estimators, 1).transpose(1, 0)
        return super().training_step((inputs, targets), batch_idx)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = torch.empty(self.n_estimators, input.size(0), 2, device=self.device)
        for n in range(self.n_estimators):
            out[n] = self.models[n](input)
        return out.transpose(0, 1)


if __name__ == "__main__":
    root = Path(__file__).parent.absolute()
    cli_train(DeepEns, RegressionDataModule, root, "deepens")
