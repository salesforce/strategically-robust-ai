from . import (
    profiling,
    remote,
    saving,
    wandb,
)

import numpy as np

def apply_concav(x, eta):
    if x <= -0.9999:
        output = -10
    elif eta == 1:
        output = np.log(x)
    else:
        output = (np.power(x + 1, 1 - eta) - 1) / (1 - eta)
    if np.isnan(output):
        return -100
    return output


