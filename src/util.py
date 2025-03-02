"""
util.py

Utility functions for the Satellite Antenna Project.
Provides memory reporting, cache clearing, padding, path loss computation, and saving populations.
"""

import os
import time
from typing import Any
import torch
import torch.nn.functional as F
import scipy.io
from config import device, wave_torch

def report_memory(device: str) -> None:
    """
    Report memory usage on GPU or MPS.

    Args:
        device (str): Device string.
    """
    if device.startswith("cuda"):
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    elif device.startswith("mps"):
        print(f"Memory allocated: {torch.mps.current_allocated_memory() / 1e6:.2f} MB")


def clear_cache(device: str) -> None:
    """
    Clear GPU or MPS cache.

    Args:
        device (str): Device string.
    """
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device.startswith("mps"):
        torch.mps.empty_cache()


def pad_to_max(mat: torch.Tensor, max_len: int, max_width: int) -> torch.Tensor:
    """
    Pad a matrix to the specified dimensions.

    Args:
        mat (torch.Tensor): Input matrix.
        max_len (int): Target number of rows.
        max_width (int): Target number of columns.

    Returns:
        torch.Tensor: Padded matrix.
    """
    if mat.shape[1] >= max_len and mat.shape[2] >= max_width:
        return mat
    pad_len = max_len - mat.shape[1]
    pad_width = max_width - mat.shape[2]
    return F.pad(mat, (0, pad_width, 0, pad_len))


@torch.jit.script
def fast_path_loss(r: torch.Tensor, wavelength: torch.Tensor = wave_torch) -> torch.Tensor:
    """
    Compute path loss using a fast approximation.

    Args:
        r (torch.Tensor): Distance tensor.
        wavelength (torch.Tensor): Wavelength tensor.

    Returns:
        torch.Tensor: Path loss in dB.
    """
    return 10 * torch.log10(4 * torch.pi * r / wavelength) + 30


def save_population_to_mat(pop: Any, filename: str = "GA_OUT.mat") -> None:
    """
    Save a population's configuration matrices to a MATLAB .mat file.

    Args:
        pop (tuple): (amplitude, phase, position, power, shape).
        filename (str): Output filename.
    """
    amplitude, phase, position, power, shape = pop
    try:
        population_data = {
            "amplitude": amplitude.cpu().numpy(),
            "phase": phase.cpu().numpy(),
            "position": position.cpu().numpy(),
            "power": power.cpu().numpy() if isinstance(power, torch.Tensor) else power,
            "shape": shape.cpu().numpy() if isinstance(shape, torch.Tensor) else shape
        }
        scipy.io.savemat(filename, population_data)
        print(f"Population saved to {filename}")
    except Exception as e:
        raise IOError(f"Failed to save population to {filename}: {e}") from e
