from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import Module
from torchmetrics import MeanAbsoluteError


class SingleRegression(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        # val metrics
        self.val_rmse = MeanAbsoluteError(squared=False)

        # test metrics
        self.test_rmse = MeanAbsoluteError(squared=False)

    @property
    def criterion(self) -> Module:
        raise NotImplementedError()

    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        return parent_parser

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore
        raise NotImplementedError()
