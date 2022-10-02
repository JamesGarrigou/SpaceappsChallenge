import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Type, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler
from pytorch_lightning.strategies import DDPStrategy

from .routine import SingleRegression


def cli_main(
    network: Type[SingleRegression],
    datamodule: Type[pl.LightningDataModule],
    root: Union[Path, str],
    net_name: str,
) -> None:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_top", type=int, default=None)
    parser.add_argument("--test", type=int, default=None)
    parser.add_argument("--profile", dest="profile", action="store_true")

    parser = pl.Trainer.add_argparse_args(parser)
    parser = datamodule.add_argparse_args(parser)
    parser = network.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.seed:
        pl.seed_everything(args.seed, workers=True)

    if args.profile:
        print(
            "Profiling will leak memory and increase the computational time."
            "Do not launch long lasting trainings with the profile flag."
        )

    if isinstance(root, str):
        root = Path(root)

    # datamodule
    args.root = str(root / args.root)
    dm = datamodule(**vars(args))

    # model
    model = network(args)

    # logger
    tb_logger = TensorBoardLogger(
        str(root / "logs"),
        name=net_name,
        default_hp_metric=False,
        log_graph=args.log_graph,
        version=args.test,
    )

    monitored = "hp/val_mse"
    mode = "min"
    # callbacks
    if args.save_top is None:
        best_checkpoint = ModelCheckpoint(
            monitor=monitored,
            mode=mode,
            save_weights_only=False,
        )
    else:
        best_checkpoint = ModelCheckpoint(
            save_top_k=args.max_epochs,
            monitor=monitored,
            mode=mode,
            save_weights_only=False,
        )
    callbacks = [best_checkpoint, LearningRateMonitor(logging_interval="step")]

    if args.profile:
        profiler = PyTorchProfiler(
            group_by_input_shapes=True,
            record_shapes=True,
            row_limit=250,
            sort_by_key="cuda_time",
        )
    else:
        profiler = None

    if args.deterministic and args.seed is None:
        print("Setting seed to 0.")
        args.__setattr__("seed", 0)

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=tb_logger,
        deterministic=(args.seed is not None),
        profiler=profiler,
    )

    if args.test is None and args.only_info is False:
        # training and testing
        trainer.fit(model, dm)
        trainer.test(datamodule=dm, ckpt_path="best")
