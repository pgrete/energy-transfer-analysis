import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
import tol_colors as tc 

def plot_transfer_map(ax, arr, norm_factor, vmax=0.1, linthresh=1e-3, title=None):
    arr_norm = arr / norm_factor

    norm = SymLogNorm(
        linthresh=linthresh,
        vmin=-vmax,
        vmax=vmax
    )

    im = ax.imshow(
        arr_norm.T,
        origin='lower',
        cmap="RdBu_r",
        aspect='auto',
        norm=norm
    )

    if title is not None:
        ax.set_title(title)

    return im