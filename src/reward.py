"""
reward.py

Reward module for evaluating antenna configurations.
Provides functions to calculate coverage percentages, average power,
and compute reward scores for each configuration.
"""

import torch
import matplotlib.pyplot as plt
from typing import Tuple
from config import flat_map_data, desired_threshold, blocked_threshold, max_dim, alpha, beta, gamma, device
from simulate import compute_received_power

def overlay_signal_strength(received_power_dBm: torch.Tensor) -> None:
    """
    Overlay received signal strength on the target map.

    Args:
        received_power_dBm (torch.Tensor): 2D tensor of received signal strength in dBm.
    """
    plt.figure(figsize=(6, 4))
    received_power_dBm = received_power_dBm.reshape_as(flat_map_data.reshape_as(flat_map_data))
    extent = [flat_map_data.min().item(), flat_map_data.max().item(), flat_map_data.min().item(), flat_map_data.max().item()]
    plt.imshow(flat_map_data.cpu(), extent=extent, aspect='equal', origin='lower', cmap='gray', alpha=0.5)
    im = plt.imshow(received_power_dBm.to("cpu"), extent=extent, aspect='equal', origin='lower', cmap='viridis', alpha=0.7)
    plt.colorbar(im, label='Received Signal Strength (dBm)', format='%.2f')
    plt.gca().invert_yaxis()
    plt.title('Satellite Signal Strength Overlay on Map Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()


def calculate_coverage_percentage(received_power_dBm: torch.Tensor) -> torch.Tensor:
    """
    Calculate coverage percentages over desired and blocked areas.

    Args:
        received_power_dBm (torch.Tensor): Tensor of received power (batch, H, W).

    Returns:
        torch.Tensor: Tensor of shape (batch, 2) with desired and blocked coverage percentages.
    """
    batch_size = received_power_dBm.shape[0]
    desired_mask = (flat_map_data == 255).float()
    blocked_mask = (flat_map_data == 0).float()

    desired_hits = ((received_power_dBm >= desired_threshold).float() * desired_mask).view(batch_size, -1).sum(dim=1)
    blocked_hits = ((received_power_dBm >= blocked_threshold).float() * blocked_mask).view(batch_size, -1).sum(dim=1)

    num_desired = desired_mask.sum()
    num_blocked = blocked_mask.sum()

    desired_coverage = desired_hits / num_desired
    blocked_coverage = blocked_hits / num_blocked

    return torch.stack([desired_coverage, blocked_coverage], dim=1)


def calculate_average_power(received_power_dBm: torch.Tensor) -> torch.Tensor:
    """
    Calculate average received power in desired and ignore areas.

    Args:
        received_power_dBm (torch.Tensor): Tensor of received power (batch, H, W).

    Returns:
        torch.Tensor: Tensor of shape (batch, 2) with average power for desired and ignore areas.
    """
    batch_size = received_power_dBm.shape[0]
    desired_mask = (flat_map_data == 255).to(received_power_dBm.dtype)
    ignore_mask = (flat_map_data == 0).to(received_power_dBm.dtype)

    desired_sum = (received_power_dBm * desired_mask).view(batch_size, -1).sum(dim=1)
    ignore_sum = (received_power_dBm * ignore_mask).view(batch_size, -1).sum(dim=1)

    num_desired = desired_mask.sum()
    num_ignore = ignore_mask.sum()

    avg_power_desired = desired_sum / num_desired
    avg_power_ignore = ignore_sum / num_ignore

    return torch.stack([avg_power_desired, avg_power_ignore], dim=-1)


def compute_reward(received_power_dBm: torch.Tensor, amplitude_matrix: torch.Tensor, power_matrix: torch.Tensor, shape: torch.Tensor, phase: torch.Tensor = None, pos: torch.Tensor = None) -> torch.Tensor:
    """
    Compute reward based on coverage percentages, average power,
    active element percentage, and various penalties.

    Args:
        received_power_dBm (torch.Tensor): Received power (batch, H, W).
        amplitude_matrix (torch.Tensor): Activation matrix (batch, H, W).
        power_matrix (torch.Tensor): Transmit power (batch, 1).
        shape (torch.Tensor): Array shape (batch, 2).

    Returns:
        torch.Tensor: Reward score for each configuration (batch,).
    """
    good_coverage = calculate_coverage_percentage(received_power_dBm)
    avg_power = calculate_average_power(received_power_dBm)
    exponent = 1.01
    norm_power = torch.pow(exponent, avg_power)
    active_percentage = torch.mean(amplitude_matrix.float(), dim=(1, 2))
    size_penalty = ((shape[:, 0] * shape[:, 1]) / (max_dim * max_dim))

    extreme_condition = ((good_coverage[:, 0] >= 0.99) & (good_coverage[:, 1] >= 0.99)) | \
                        ((good_coverage[:, 0] <= 0.01) & (good_coverage[:, 1] <= 0.01))
    extreme_reward = norm_power[:, 0] - norm_power[:, 1] - 3.0
    normal_reward = (good_coverage[:, 0] * norm_power[:, 0] - good_coverage[:, 1] * norm_power[:, 1]) * alpha - \
                    active_percentage * gamma - size_penalty * gamma - power_matrix.squeeze(1) * beta

    reward = torch.where(extreme_condition, extreme_reward, normal_reward)
    reward = torch.where(torch.isnan(reward) | torch.isinf(reward), torch.tensor(-100.0, device=reward.device), reward)
    return reward


def compute_reward_vector(population: tuple, device: str = device) -> torch.Tensor:
    """
    Compute reward for each configuration in a population.

    Args:
        population (tuple): (amplitude_mats, phase_mats, position_vectors, power, shapes).
        device (str): Device for tensor operations.

    Returns:
        torch.Tensor: Reward vector of shape (num_configs,).
    """
    amplitude_mats, phase_mats, position_vectors, power, shapes = population
    num_configs = power.shape[0]
    reward_vector = torch.zeros((num_configs), device=device)

    for i in range(num_configs):
        rp = compute_received_power(amplitude_mats[i], phase_mats[i], position_vectors[i], power[i], shapes[i], debug=False, device=device)
        reward_vector[i] = compute_reward(rp.unsqueeze(0), amplitude_mats[i].unsqueeze(0), power[i].unsqueeze(0), shapes[i].unsqueeze(0))[0]

    return reward_vector
