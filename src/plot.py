"""
plot.py

Plotting module for visualizing antenna array configurations and signal strength.
Provides functions to overlay signal strength on map data and plot the array factor.
"""

import math
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional
from config import lons, lats, map_data, device
from simulate import compute_received_power, compute_array_factor
from reward import calculate_coverage_percentage, calculate_average_power
from transform import sat_centered_spherical
from util import save_population_to_mat

def overlay_signal_strength(received_power_dBm: torch.Tensor) -> None:
    """
    Overlay the received signal strength on the target map.

    Args:
        received_power_dBm (torch.Tensor): 2D tensor of received signal strength in dBm.
    """
    plt.figure(figsize=(6, 4))
    received_power_dBm = received_power_dBm.reshape_as(map_data)
    extent = [lons[0, 0].item(), lons[0, -1].item(), lats[0, 0].item(), lats[-1, 0].item()]
    plt.imshow(map_data, extent=extent, aspect='equal', origin='lower', cmap='gray', alpha=0.5)
    im = plt.imshow(received_power_dBm.to("cpu"), extent=extent, aspect='equal', origin='lower', cmap='viridis', alpha=0.7)
    plt.colorbar(im, label='Received Signal Strength (dBm)', format='%.2f')
    plt.gca().invert_yaxis()
    plt.title('Satellite Signal Strength Overlay on Map Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()


def plot_best_configuration(population: tuple, reward_vector: torch.Tensor, name: Optional[str] = None, device: str = device) -> None:
    """
    Plot the best performing configuration from the population.

    Args:
        population (tuple): (amplitudes, phases, positions, powers, shapes).
        reward_vector (torch.Tensor): Reward scores for each configuration.
        name (Optional[str]): Filename to save population data.
        device (str): Device for tensor operations.
    """

    best_reward, best_index = torch.topk(reward_vector, 1)
    best_config = tuple(tensor[best_index] for tensor in population)
    best_index: int = int(torch.argmax(reward_vector).item())
    best_amplitude, best_phase, best_position, best_power, best_shape = best_config

    plot_array_factor(best_amplitude, best_phase, best_position, best_shape, plot_combined=True, plot_map=True, plot_separate=True, device=device)

    received_power_dBm = compute_received_power(best_amplitude, best_phase, best_position, best_power, best_shape, device=device)
    if name:
        save_population_to_mat((best_amplitude, best_phase, best_position, best_power, best_shape), name)
    best_power_linear = 10 ** (best_power / 10) / 1000
    overlay_signal_strength(received_power_dBm)
    print(f"Best configuration index: {best_index}, Reward: {best_reward.item()}")

    coverage = calculate_coverage_percentage(received_power_dBm) * 100
    avg_power = calculate_average_power(received_power_dBm)
    array_active_percentage = torch.mean(best_amplitude.float()).item() * 100
    print(f"Target area above threshold: {coverage[:, 0].item():.4f}% Average Target Area dBm: {avg_power[:, 0].item():.4f}")
    print(f"Exclusion area above threshold: {coverage[:, 1].item():.4f}% Average Exclusion Area dBm: {avg_power[:, 1].item():.4f}")
    print(f"Array Active Percentage: {array_active_percentage:.2f}%")
    print(f"Transmit Power: {best_power.item()} dBm, {best_power_linear.item()} W")
    print(f"Array Dimensions: {best_shape.tolist()}")
    print(f"Position: {best_position.tolist()}")

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device.startswith("mps"):
        torch.mps.empty_cache()


def plot_array_factor(amplitude_matrix: torch.Tensor, phase_matrix: torch.Tensor, position_vector: torch.Tensor, shape: torch.Tensor, plot_map: bool = True, plot_combined: bool = True, plot_separate: bool = True, device: str = device) -> None:
    """
    Plot the array factor as a function of azimuth and zenith angles.

    Args:
        amplitude_matrix (torch.Tensor): Binary activation matrix.
        phase_matrix (torch.Tensor): Phase matrix for the array.
        position_vector (torch.Tensor): Satellite position.
        shape (torch.Tensor): Array shape (dimensions).
        plot_map (bool): Whether to overlay a rectangular region based on satellite position.
        plot_combined (bool): Whether to show a 2D plot of the array factor.
        plot_separate (bool): Whether to show separate 1D cuts.
        device (str): Device for tensor operations.
    """
    zenith_deg = torch.linspace(0, 180, 181, device=device)
    azimuth_deg = torch.linspace(-180, 180, 361, device=device)
    zenith_rad = torch.deg2rad(zenith_deg)
    azimuth_rad = torch.deg2rad(azimuth_deg)
    zenith_grid, azimuth_grid = torch.meshgrid(zenith_rad, azimuth_rad, indexing="ij")

    AF = compute_array_factor(azimuth_grid.flatten().unsqueeze(0), zenith_grid.flatten().unsqueeze(0),
                              amplitude_matrix, phase_matrix, shape, device=device)
    AF_dBm = 20 * torch.log10(torch.abs(AF) + 1e-12)
    AF_dBm_np = AF_dBm.reshape(zenith_grid.shape).cpu().detach().numpy()

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 4))
    if plot_combined:
        im = ax0.imshow(AF_dBm_np, extent=[-math.pi, math.pi, 0, math.pi],
                        aspect="auto", cmap="jet", origin="lower")
        fig.colorbar(im, ax=ax0, label="Array Factor (dBm)")
        ax0.set_xlabel("Azimuth φ (radians)")
        ax0.set_ylabel("Zenith θ (radians)")
        ax0.set_title("Array Factor Pattern")
        if plot_map:
            r, az, zn = sat_centered_spherical(position_vector).unbind(dim=2)
            width = torch.max(az) - torch.min(az)
            height = torch.max(zn) - torch.min(zn)
            rect = Rectangle((torch.min(az).cpu(), torch.min(zn).cpu()), width.cpu(), height.cpu(),
                             edgecolor='purple', facecolor='none', lw=2)
            ax0.add_patch(rect)

    if plot_separate:
        fixed_zenith_idx = zenith_deg.shape[0] // 2
        fixed_azimuth_idx = azimuth_deg.shape[0] // 2
        fixed_zenith_deg_val = zenith_deg[fixed_zenith_idx].item()
        fixed_azimuth_deg_val = azimuth_deg[fixed_azimuth_idx].item()

        azimuth_cut_dBm = AF_dBm_np[fixed_zenith_idx, :]
        ax1.plot(azimuth_deg.cpu().numpy(), azimuth_cut_dBm, 'b-', lw=2)
        ax1.set_xlabel("Azimuth (deg)")
        ax1.set_ylabel("Array Factor (dBm)")
        ax1.set_title(f"Azimuth Cut at zenith = {fixed_zenith_deg_val:.1f}°")
        r, az, zn = sat_centered_spherical(position_vector).unbind(dim=2)
        az = az * (180 / math.pi)
        ax1.axvspan(torch.min(az).cpu(), torch.max(az).cpu(), color='red', alpha=0.3, label='Azimuth Region')
        ax1.legend()

        zenith_cut_dBm = AF_dBm_np[:, fixed_azimuth_idx]
        ax2.plot(zenith_deg.cpu().numpy(), zenith_cut_dBm, 'g-', lw=2)
        ax2.set_xlabel("Zenith (deg)")
        ax2.set_ylabel("Array Factor (dBm)")
        ax2.set_title(f"Zenith Cut at azimuth = {fixed_azimuth_deg_val:.1f}°")

    plt.tight_layout()
    plt.show()
