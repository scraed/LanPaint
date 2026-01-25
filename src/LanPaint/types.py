from typing import NamedTuple, Optional

import torch


class LangevinState(NamedTuple):
    v: Optional[torch.Tensor]
    C: Optional[torch.Tensor]
    x0: Optional[torch.Tensor]

