"""
transform.py

Coordinate transformation module.
Converts geodetic coordinates (LLA) to ECEF and computes satellite-centered spherical coordinates.
"""

import math
from typing import Tuple
import torch
from config import *

def lla_to_ecef(lla: torch.Tensor, device: str = device) -> torch.Tensor:
    """
    Convert geodetic coordinates (latitude, longitude, altitude) to ECEF coordinates.

    Args:
        lla (torch.Tensor): Tensor of shape (batch, 3) with (lat, lon, alt).
        device (str): Device for tensor operations.

    Returns:
        torch.Tensor: ECEF coordinates of shape (batch, 3).
    """
    try:
        lat, lon, alt = lla[:, 0], lla[:, 1], lla[:, 2]
    except IndexError as e:
        lla = lla.unsqueeze(0)
        lat, lon, alt = lla[:, 0], lla[:, 1], lla[:, 2]

    lat = torch.deg2rad(lat)
    lon = torch.deg2rad(lon)

    a: float = 6378137.0  # Semi-major axis (m)
    e2: float = 6.69437999014e-3  # Square of first eccentricity

    sin_lat = torch.sin(lat)
    cos_lat = torch.cos(lat)
    N = a / torch.sqrt(1 - e2 * sin_lat ** 2)

    x = (N + alt) * cos_lat * torch.cos(lon)
    y = (N + alt) * cos_lat * torch.sin(lon)
    z = (N * (1 - e2) + alt) * sin_lat

    return torch.stack((x, y, z), dim=-1)


observations_ecef = lla_to_ecef(observations)


def sat_centered_spherical(satellite_pos: torch.Tensor, directed: bool = True, device: str = device) -> torch.Tensor:
    """
    Convert ECEF coordinates to satellite-centered spherical coordinates.

    Args:
        satellite_pos (torch.Tensor): Satellite position in LLA format (batch, 3).
        directed (bool): If True, center azimuth and adjust zenith.
        device (str): Device for tensor operations.

    Returns:
        torch.Tensor: Spherical coordinates (batch, num_observations, 3) with (r, azimuth, zenith).
    """
    batch_size = satellite_pos.shape[0]
    satellite_ecef = lla_to_ecef(satellite_pos.to(device))
    satellite_ecef = satellite_ecef.unsqueeze(1)

    # Assuming observations_ecef is defined elsewhere
    observations_relative = observations_ecef.unsqueeze(0) - satellite_ecef
    x, y, z = observations_relative[..., 0], observations_relative[..., 1], observations_relative[..., 2]
    r = torch.linalg.norm(observations_relative, dim=-1)
    r_2d = torch.linalg.norm(observations_relative[..., :2], dim=-1)

    azimuth = torch.pi - torch.atan2(x, y)
    zenith = torch.atan2(z, r_2d)
    azimuth = azimuth - azimuth.mean(dim=1, keepdim=True)
    if directed:
        zenith = -(zenith - zenith.mean(dim=1, keepdim=True)) + torch.pi / 2
    else:
        zenith = torch.pi / 2 - zenith
    if azimuth.isnan().any() or zenith.isnan().any():
        print("⚠️ NAN VALUE DETECTED")
    return torch.stack((r, azimuth, zenith), dim=2)
