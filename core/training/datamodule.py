from argparse import ArgumentParser
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Any, List, Optional, Union
from pathlib import Path


class RegressionDataModule(LightningDataModule):
    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int,
        val_split: int = 0.2,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if isinstance(root, str):
            root = Path(root)

        self.root = root
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.split_number = 0
        self.n_splits = None

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage)

    def train_dataloader(self) -> DataLoader:
        return super().train_dataloader()

    def val_dataloader(self) -> DataLoader:
        return super().val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return super().test_dataloader()

    @classmethod
    def add_argparse_args(
        cls,
        parent_parser: ArgumentParser,
        **kwargs: Any,
    ) -> ArgumentParser:
        p = parent_parser.add_argument_group("datamodule")
        p.add_argument("--root", type=str, default="./data/")
        p.add_argument("--batch_size", type=int, default=128)
        p.add_argument("--num_workers", type=int, default=4)
        return parent_parser
