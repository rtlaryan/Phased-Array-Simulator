"""
generate.py

Module for generating and mutating antenna array configurations.
Provides functions to create random populations, apply phase tapers,
and combine or mutate populations for evolutionary algorithms.
"""

import math
from typing import Tuple
import torch
from config import *
from util import clear_cache

# Define type for population tuple
Population = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

def skewed_randint(count: int, min_dim: int = min_dim, max_dim: int = max_dim, alpha: float = 2.0, beta: float = 4.0, device: str = device) -> torch.Tensor:
    """
    Generate skewed random integers using a Beta distribution scaled to the specified range.

    Args:
        count (int): Number of random integer pairs to generate.
        min_dim (int): Minimum value for each integer.
        max_dim (int): Maximum value for each integer.
        alpha (float): Alpha parameter for the Beta distribution.
        beta (float): Beta parameter for the Beta distribution.
        device (str): Device for tensor storage.

    Returns:
        torch.Tensor: Tensor of shape (count, 2) with skewed random integers.
    """
    beta_samples = torch.distributions.Beta(alpha, beta).sample((count, 2)).to(device)
    scaled_samples = min_dim + (max_dim - min_dim) * beta_samples
    return scaled_samples.round().to(torch.int8)


def generate_radial_phase_taper(count: int, shape: Tuple[int, int], noise: float = 0.0, invert: bool = False, center: bool = False, dtype: torch.dtype = torch.float32, device: str = device) -> torch.Tensor:
    """
    Generate phase matrices with a radial taper centered at a random (or fixed) point.

    Args:
        count (int): Number of matrices to generate.
        shape (Tuple[int, int]): (L, W) specifying matrix dimensions.
        noise (float): Standard deviation of Gaussian noise added to the phase.
        invert (bool): If True, invert the taper.
        center (bool): If True, use the center of the matrix as the taper center.
        dtype (torch.dtype): Data type for the tensor.
        device (str): Device for tensor storage.

    Returns:
        torch.Tensor: Tensor of shape (count, L, W) with phase values in [0, 2*pi].
    """
    L, W = shape
    yy, xx = torch.meshgrid(torch.arange(L, dtype=dtype, device=device),
                             torch.arange(W, dtype=dtype, device=device),
                             indexing='ij')
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    if center:
        centers_x = torch.full((count, 1, 1), W / 2, dtype=dtype, device=device)
        centers_y = torch.full((count, 1, 1), L / 2, dtype=dtype, device=device)
    else:
        centers_x = torch.rand(count, 1, 1, dtype=dtype, device=device) * (W - 1)
        centers_y = torch.rand(count, 1, 1, dtype=dtype, device=device) * (L - 1)

    dist = torch.sqrt((xx - centers_x) ** 2 + (yy - centers_y) ** 2)
    d1 = torch.sqrt(centers_x ** 2 + centers_y ** 2)
    d2 = torch.sqrt((centers_x - (W - 1)) ** 2 + centers_y ** 2)
    d3 = torch.sqrt(centers_x ** 2 + (centers_y - (L - 1)) ** 2)
    d4 = torch.sqrt((centers_x - (W - 1)) ** 2 + (centers_y - (L - 1)) ** 2)
    max_d = torch.max(torch.cat([d1, d2, d3, d4], dim=0), dim=0)[0].unsqueeze(0)

    phase = torch.where(max_d > 0,
                        2 * torch.pi * ((1 - invert) - (1 - 2 * invert) * dist / max_d),
                        2 * torch.pi * torch.ones_like(dist))
    if noise != 0:
        phase = phase + noise * torch.randn_like(phase, dtype=dtype, device=device)
    phase = torch.remainder(phase, 2 * torch.pi)
    return phase


def taper_phase_line(length: int, count: int, invert: bool, center: bool, dtype: torch.dtype, device: str) -> torch.Tensor:
    """
    Create a 1D phase taper for a given length.

    Args:
        length (int): Length of the phase line.
        count (int): Number of phase lines to generate.
        invert (bool): If True, invert the taper.
        center (bool): If True, use the center of the line as the peak.
        dtype (torch.dtype): Data type.
        device (str): Device for tensor storage.

    Returns:
        torch.Tensor: Tensor of shape (count, length) representing phase tapers.
    """
    coords = torch.arange(length, dtype=dtype, device=device).unsqueeze(0)
    if center:
        centers = torch.full((count, 1), length / 2, dtype=dtype, device=device)
    else:
        centers = torch.rand(count, 1, dtype=dtype, device=device) * (length - 1)
    dist = torch.abs(coords - centers)
    max_dist = torch.max(centers, (length - 1) - centers)
    phase_line = torch.where(max_dist > 0,
                             2 * torch.pi * ((1 - invert) - (1 - 2 * invert) * dist / max_dist),
                             2 * torch.pi * torch.ones_like(dist, device=device))
    return phase_line


def generate_linear_phase_taper(count: int, shape: Tuple[int, int], noise: float = 0.0, invert: bool = False, center: bool = False, axis: int = 1, dtype: torch.dtype = torch.float32, device: str = device) -> torch.Tensor:
    """
    Generate phase matrices with a linear taper along a specified axis.

    Args:
        count (int): Number of matrices to generate.
        shape (Tuple[int, int]): (L, W) for matrix dimensions.
        noise (float): Standard deviation of Gaussian noise added.
        invert (bool): If True, invert the taper.
        center (bool): If True, use the center of the line as peak.
        axis (int): 0 for vertical, 1 for horizontal, 2 for random axis selection.
        dtype (torch.dtype): Data type.
        device (str): Device for tensor storage.

    Returns:
        torch.Tensor: Tensor of shape (count, L, W) with phase values.
    """
    L, W = shape

    def phase_for_axis(chosen_axis: int, invert: bool, center: bool) -> torch.Tensor:
        if chosen_axis == 1:
            phase_line = taper_phase_line(W, count, invert, center, dtype, device)
            return phase_line.unsqueeze(1).expand(count, L, W)
        elif chosen_axis == 0:
            phase_line = taper_phase_line(L, count, invert, center, dtype, device)
            return phase_line.unsqueeze(2).expand(count, L, W)
        else:
            raise ValueError("Invalid axis value in phase_for_axis.")

    if axis in (0, 1):
        phase_matrix = phase_for_axis(axis, invert, center)
    elif axis == 2:
        phases_h = phase_for_axis(1, invert, center)
        phases_v = phase_for_axis(0, invert, center)
        rand_bool = torch.randint(0, 2, (count,), device=device, dtype=torch.bool)
        mask = rand_bool.view(count, 1, 1)
        phase_matrix = torch.where(mask, phases_h, phases_v)
    else:
        raise ValueError("axis must be 0 (vertical), 1 (horizontal), or 2 (randomized per matrix)")

    if noise:
        phase_matrix = phase_matrix + noise * torch.randn_like(phase_matrix, dtype=dtype)
    phase_matrix = torch.remainder(phase_matrix, 2 * torch.pi)
    return phase_matrix


def generate_beamforming_phase(count: int, shape: Tuple[int, int], azimuth: float = 0, zenith: float = math.pi/2, noise: float = phase_mutation_rate, dtype: torch.dtype = torch.float32, device: str = device) -> torch.Tensor:
    """
    Generate phase matrices for beam steering in a planar array.

    Args:
        count (int): Number of matrices to generate.
        shape (Tuple[int, int]): (L, W) for matrix dimensions.
        azimuth (float): Base azimuth angle in radians.
        zenith (float): Base zenith angle in radians.
        noise (float): Noise factor for random perturbations.
        dtype (torch.dtype): Data type.
        device (str): Device for tensor storage.

    Returns:
        torch.Tensor: Tensor of shape (count, L, W) with beamforming phase values.
    """
    length, width = shape
    azimuth_tensor = torch.tensor(azimuth, dtype=dtype, device=device) + (torch.rand(count, device=device, dtype=dtype) * 2 - 1) * noise
    zenith_tensor = torch.tensor(zenith, dtype=dtype, device=device) + (torch.rand(count, device=device, dtype=dtype) * 2 - 1) * noise

    X, Y = torch.meshgrid(torch.arange(length, device=device, dtype=dtype),
                          torch.arange(width, device=device, dtype=dtype), indexing="ij")
    X = X.unsqueeze(0)
    Y = Y.unsqueeze(0)
    azimuth_tensor = azimuth_tensor.view(count, 1, 1)
    zenith_tensor = zenith_tensor.view(count, 1, 1)

    phase_matrices = -k * d * (X * torch.sin(zenith_tensor) * torch.cos(azimuth_tensor) +
                               Y * torch.sin(zenith_tensor) * torch.sin(azimuth_tensor))
    phase_matrices.remainder_(2 * torch.pi)
    return phase_matrices


def randomize_positions(count: int, mutation_rate: float = mutation_rate, dtype: torch.dtype = torch.float32, device: str = device) -> torch.Tensor:
    """
    Generate random satellite positions with constrained longitudinal variation.

    Args:
        count (int): Number of positions to generate.
        mutation_rate (float): Scaling factor for random perturbation.
        dtype (torch.dtype): Data type.
        device (str): Device for tensor storage.

    Returns:
        torch.Tensor: Tensor of shape (count, 3) with positions.
    """
    satellite_longitudes = start_sat_pos[1] + torch.empty(count, dtype=dtype, device=device).uniform_(-dist_range, dist_range) * mutation_rate
    satellite_latitudes = sat_latitude.expand(count)
    satellite_altitudes = start_sat_pos[2].expand(count)
    return torch.stack((satellite_latitudes, satellite_longitudes, satellite_altitudes), dim=1).to(dtype=dtype)


def generate_random_population(count: int, mutation_rate: float = mutation_rate, dtype: torch.dtype = torch.float32, device: str = device) -> Population:
    """
    Generate a random population of antenna configurations.

    Args:
        count (int): Number of configurations to generate.
        mutation_rate (float): Mutation rate for position.
        dtype (torch.dtype): Data type.
        device (str): Device for tensor storage.

    Returns:
        Population: (amplitudes, phases, positions, powers, shapes)
    """
    shapes = skewed_randint(count, min_dim, max_dim, device=device)
    max_len, max_width = shapes.max(dim=0).values
    rows_expanded = torch.arange(max_len, device=device).view(1, max_len, 1).expand(count, -1, -1)
    cols_expanded = torch.arange(max_width, device=device).view(1, 1, max_width).expand(count, -1, -1)
    len_mask = rows_expanded < shapes[:, 0].view(count, 1, 1)
    width_mask = cols_expanded < shapes[:, 1].view(count, 1, 1)
    mask = len_mask & width_mask

    amplitudes = torch.randint(0, 2, (count, max_len, max_width), dtype=dtype, device=device)
    amplitudes *= mask.int()

    phases = torch.rand((count, max_len, max_width), dtype=dtype, device=device) * max_phase
    phases *= mask.int()

    positions = randomize_positions(count, mutation_rate=mutation_rate, device=device)
    powers = torch.randint(min_power, max_power + 1, (count, 1), dtype=torch.float32, device=device)

    return amplitudes, phases, positions, powers, shapes


def generate_uniform_population(count: int, dtype: torch.dtype = torch.float32, device: str = device) -> Population:
    """
    Generate a uniform population of configurations with fixed structure.

    Args:
        count (int): Number of configurations to generate.
        dtype (torch.dtype): Data type.
        device (str): Device for tensor storage.

    Returns:
        Population: (amplitudes, phases, positions, powers, shapes)
    """
    shapes = skewed_randint(count, min_dim, max_dim, device=device)
    max_len, max_width = shapes.max(dim=0).values
    rows_expanded = torch.arange(max_len, device=device).view(1, max_len, 1).expand(count, -1, -1)
    cols_expanded = torch.arange(max_width, device=device).view(1, 1, max_width).expand(count, -1, -1)
    len_mask = rows_expanded < shapes[:, 0].view(count, 1, 1)
    width_mask = cols_expanded < shapes[:, 1].view(count, 1, 1)
    mask = len_mask & width_mask

    amplitudes = torch.ones((count, max_len, max_width), dtype=dtype, device=device)
    amplitudes *= mask.int()

    phases = torch.zeros((count, max_len, max_width), dtype=dtype, device=device)
    phases *= mask.int()

    positions = randomize_positions(count, mutation_rate=0, device=device)
    powers = torch.ones((count, 1), dtype=torch.float32, device=device) * base_power

    return amplitudes, phases, positions, powers, shapes


def generate_population(count: int, phase_type: str = None, invert_phase: bool = False, dtype: torch.dtype = torch.float32, device: str = device) -> Population:
    """
    Generate a population with specified phase type.

    Args:
        count (int): Number of configurations to generate.
        phase_type (str): Type of phase pattern ("linear", "radial", "beamforming", "random", "mixed").
        invert_phase (bool): If True, invert the phase taper.
        dtype (torch.dtype): Data type.
        device (str): Device for tensor storage.

    Returns:
        Population: (amplitudes, phases, positions, powers, shapes)
    """
    shapes = skewed_randint(count, min_dim, max_dim, device=device)
    max_len, max_width = shapes.max(dim=0).values
    rows_expanded = torch.arange(max_len, device=device).view(1, max_len, 1).expand(count, -1, -1)
    cols_expanded = torch.arange(max_width, device=device).view(1, 1, max_width).expand(count, -1, -1)
    len_mask = rows_expanded < shapes[:, 0].view(count, 1, 1)
    width_mask = cols_expanded < shapes[:, 1].view(count, 1, 1)
    mask = len_mask & width_mask

    amplitudes = torch.ones((count, max_len, max_width), dtype=dtype, device=device)
    amplitudes *= mask.int()

    if phase_type == "linear":
        phases = generate_linear_phase_taper(count, (max_len, max_width), invert=invert_phase, noise=phase_mutation_rate, dtype=dtype, device=device)
    elif phase_type == "radial":
        phases = generate_radial_phase_taper(count, (max_len, max_width), invert=invert_phase, noise=phase_mutation_rate, dtype=dtype, device=device)
    elif phase_type == "beamforming":
        phases = generate_beamforming_phase(count, (max_len, max_width), noise=phase_mutation_rate, dtype=dtype, device=device)
    elif phase_type == "random":
        phases = torch.rand((count, max_len, max_width), dtype=dtype, device=device) * 2 * torch.pi
    elif phase_type == "mixed":
        if count >= 4:
            size, remainder = divmod(count, 4)
            phases = torch.cat((
                generate_linear_phase_taper(size, (max_len, max_width), invert=invert_phase, noise=phase_mutation_rate, dtype=dtype, device=device),
                generate_radial_phase_taper(size, (max_len, max_width), invert=invert_phase, noise=phase_mutation_rate, dtype=dtype, device=device),
                generate_beamforming_phase(size, (max_len, max_width), noise=phase_mutation_rate, dtype=dtype, device=device),
                torch.rand((size, max_len, max_width), dtype=dtype, device=device) * 2 * torch.pi,
                torch.zeros((remainder, max_len, max_width), dtype=dtype, device=device)
            ), dim=0)
        else:
            phases = torch.rand((count, max_len, max_width), dtype=dtype, device=device) * 2 * torch.pi
    else:
        raise ValueError("Invalid Phase Type. Use 'linear', 'radial', 'beamforming', 'random', or 'mixed'.")

    phases *= mask.int()
    positions = randomize_positions(count, device=device)
    powers = torch.randint(min_power, max_power + 1, (count, 1), dtype=torch.float32, device=device)

    return amplitudes, phases, positions, powers, shapes


def combine_populations(*populations: Population) -> Population:
    """
    Combine multiple populations into a single population.

    Args:
        populations: Multiple tuples of (amplitudes, phases, positions, powers, shapes).

    Returns:
        Population: Combined (amplitudes, phases, positions, powers, shapes).
    """
    total_count: int = sum(pop[0].shape[0] for pop in populations)
    global_max_len: int = max(pop[0].shape[1] for pop in populations)
    global_max_width: int = max(pop[0].shape[2] for pop in populations)

    ref_dtype = populations[0][0].dtype
    ref_device = populations[0][0].device

    combined_amplitudes = torch.zeros((total_count, global_max_len, global_max_width), dtype=ref_dtype, device=ref_device)
    combined_phases = torch.zeros((total_count, global_max_len, global_max_width), dtype=ref_dtype, device=ref_device)

    positions_list = []
    powers_list = []
    shapes_list = []

    current_index = 0
    for pop in populations:
        amps, phases, positions, powers, shapes = pop
        count = amps.shape[0]
        L, W = amps.shape[1], amps.shape[2]
        combined_amplitudes[current_index:current_index + count, :L, :W] = amps
        combined_phases[current_index:current_index + count, :L, :W] = phases
        positions_list.append(positions)
        powers_list.append(powers)
        shapes_list.append(shapes)
        current_index += count

    combined_positions = torch.cat(positions_list, dim=0)
    combined_powers = torch.cat(powers_list, dim=0)
    combined_shapes = torch.cat(shapes_list, dim=0)

    return combined_amplitudes, combined_phases, combined_positions, combined_powers, combined_shapes


def mutate_population(old_population: Population, device: str = device) -> Population:
    """
    Apply mutations to a population, including shape, amplitude, phase, position, and power mutations.

    Args:
        old_population (Population): (amplitudes, phases, positions, powers, shapes).
        device (str): Device for tensor storage.

    Returns:
        Population: Mutated population (amplitudes, phases, positions, powers, shapes).
    """
    n = old_population[0].shape[0]
    population = [t.repeat_interleave(num_mutations, dim=0) for t in old_population]
    amplitudes, phases, positions, powers, shapes = population

    shape_modifications = torch.randint(-1, 2, shapes.shape, device=device)
    mutation_mask = torch.rand_like(shapes.float()) < shape_mutation_rate
    new_shapes = torch.clamp(shapes + mutation_mask * shape_modifications, min=min_dim, max=max_dim)

    new_max_len: int = int(new_shapes[:, 0].max().item())
    new_max_width: int = int(new_shapes[:, 1].max().item())

    padded_amps = torch.zeros((n * num_mutations, new_max_len, new_max_width), device=device)
    padded_phases = torch.zeros((n * num_mutations, new_max_len, new_max_width), device=device)

    for i in range(n * num_mutations):
        orig_len: int = int(shapes[i, 0].item())
        orig_width: int = int(shapes[i, 1].item())
        mut_len: int = int(new_shapes[i, 0].item())
        mut_width: int = int(new_shapes[i, 1].item())
        copy_len: int = min(orig_len, mut_len)
        copy_width: int = min(orig_width, mut_width)
        amp_mask = torch.rand((copy_len, copy_width), device=device)
        padded_amps[i, :copy_len, :copy_width] = torch.where(
            amp_mask < amp_mutation_rate,
            1 - amplitudes[i, :copy_len, :copy_width],
            amplitudes[i, :copy_len, :copy_width]
        )
        padded_phases[i, :copy_len, :copy_width] = torch.remainder(
            (phases[i, :copy_len, :copy_width] + torch.randn((copy_len, copy_width), device=device) * 2 * torch.pi * phase_mutation_rate),
            2 * torch.pi
        )

    positions[:, 1].add_(torch.randn_like(positions[:, 1]) * position_mutation_rate * dist_range)
    powers.add_(torch.randn_like(powers) * power_mutation_rate)
    
    return padded_amps, padded_phases, positions, powers, new_shapes


def evolve_population(population: Population, rewards: torch.Tensor, device: str = device) -> Population:
    """
    Evolve the population using selection, mutation, and the addition of random/uniform configurations.

    Args:
        population (Population): (amplitudes, phases, positions, powers, shapes).
        rewards (torch.Tensor): Reward scores for each configuration.
        device (str): Device for tensor operations.

    Returns:
        Population: Evolved population.
    """
    topk = max(num_survivors, 1)
    _, indices = torch.topk(rewards, topk)
    survivors = tuple(tensor[indices] for tensor in population)
    mutated = mutate_population(survivors, device=device)
    random_pop = generate_random_population(num_random, mutation_rate=mutation_rate, dtype=torch.float32, device=device)
    uniform_pop = generate_uniform_population(num_uniform, dtype=torch.float32, device=device)
    combined = combine_populations(survivors, mutated, random_pop, uniform_pop)
    return combined
