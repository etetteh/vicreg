import os
import torch
import random
import numpy as np

def set_seed(seed: int = 6):
    """
    This function controls sources of randomness to aid reproducibility.
    Args:
        seed int: to initialize random number generator (RNG)
    """
    os.environ["GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def set_worker_seed(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
