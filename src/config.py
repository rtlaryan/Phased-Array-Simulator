"""
config.py

Configuration file for the Satellite Antenna Project.
Sets global parameters, constants, and loads the target map data.
"""

import os
import json
import time
import platform
import math
import random
import torch
import scipy.io
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython.display import clear_output
from matplotlib.patches import Rectangle
from typing import Any

# Device selection for GPU acceleration
device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Assignment parameters
frequency: float = 30e9          # Operating frequency in Hz
fc: float = 30e9                 # Center frequency (Hz)
c: float = 3e8                   # Speed of light (m/s)
wavelength: float = c / fc       # Wavelength (m)
wave_torch: torch.Tensor = torch.tensor(wavelength, device=device, dtype=torch.float32)

geo_altitude: float = 36000e3    # Geostationary orbit altitude (m)
earth_radius: float = 6371e3     # Earth's radius (m)
k: torch.Tensor = 2 * torch.pi / wavelength  # Wave number

# Load map data from MATLAB file with error handling
try:
    mat_data: Any = scipy.io.loadmat('GAmap.mat')
except FileNotFoundError as e:
    raise FileNotFoundError("GAmap.mat not found in the working directory.") from e
except Exception as e:
    raise Exception("Error loading GAmap.mat: " + str(e)) from e

map_data: torch.Tensor = torch.tensor(mat_data['GA']['data'][0][0], dtype=torch.float32, device="cpu")
flat_map_data: torch.Tensor = map_data.flatten().to(device)
valid_count: torch.Tensor = flat_map_data.sum().clamp(min=1).unsqueeze(0)  # Prevent division by zero

lats: torch.Tensor = -1 * torch.tensor(mat_data['GA']['lat'][0][0], dtype=torch.float32, device=device)
lons: torch.Tensor = torch.tensor(mat_data['GA']['lon'][0][0], dtype=torch.float32, device=device)

# Generate observation points from latitudes and longitudes (with zero elevation)
observations: torch.Tensor = torch.stack([lats.flatten(), lons.flatten(), torch.zeros_like(lats.flatten())]).to(device).T
num_obs: int = lats.flatten().shape[0]

# Fixed array parameters
base_power: torch.Tensor = torch.tensor(60, device=device)  # Base transmission power
G_rx_dBm: int = 10                                        # Receiver gain in dBm
d: float = wavelength / 2                                # Element spacing

# Variable array parameters
max_power: int = 70   # Maximum transmission power (dBm)
min_power: int = 1   # Minimum transmission power (dBm)
max_dim: int = 25     # Maximum array dimension
min_dim: int = 8      # Minimum array dimension
max_phase: torch.Tensor = 2 * torch.pi  # Maximum phase (radians)

# Evolution parameters
num_survivors: int = 3
num_random: int = 10
num_uniform: int = 10
num_mutations: int = 15
dist_range: float = 2  # Longitude range for satellite position mutations
mutation_rate: float = 0.1
phase_mutation_rate: float = 0.1
shape_mutation_rate: float = 0.05
amp_mutation_rate: float = 0.05
power_mutation_rate: int = 3
position_mutation_rate: float = 0.05

# Reward parameters
alpha: float = 1
beta: float = 0.1
gamma: float = 0.01
desired_threshold: int = -80  # Desired signal threshold (dB)
blocked_threshold: int = -80  # Blocked area threshold (dB)

# Training parameters
batch_size: int = 10
af_batch_size: int = 1
max_evolutions: int = 100000
report_evolutions: int = 10
save_evolutions: int = 50

# Base satellite position for geostationary orbit (latitude set to 0)
sat_latitude: torch.Tensor = torch.nanmean(lats, dtype=torch.float32) * 0
start_sat_pos: torch.Tensor = torch.stack([
    sat_latitude,
    torch.nanmean(lons, dtype=torch.float32),
    torch.tensor(geo_altitude, dtype=torch.float32, device=device)
], dim=0).to(device)
