from typing import Optional

import torch

def _create_generator(seed: Optional[int] = None) -> torch.Generator:
    generator = torch.Generator()
    if seed is None:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator.manual_seed(seed)
    return generator
