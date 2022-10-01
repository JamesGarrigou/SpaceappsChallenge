from pathlib import Path
from typing import Any, Union

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class SolarDataset(Dataset):
    def __init__(
        self,
        root: Union[Path, str],
    ) -> None:
        super().__init__()

        if isinstance(root, str):
            root = Path(root)
        self.root = root

        self.features = torch.as_tensor(0)

    def __getitem__(self, index: Any) -> T_co:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return self.features.shape[0]
