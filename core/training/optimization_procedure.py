import numpy as np
import torch.nn as nn
import torch.optim as optim


def optimizer(model: nn.Module) -> dict:
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.05,
    )
    # scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=[25, 50],
    #     gamma=0.1,
    # )
    return {"optimizer": optimizer}  # , "lr_scheduler": scheduler}
