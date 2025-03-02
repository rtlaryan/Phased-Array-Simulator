"""
evolve.py

Evolution module for optimizing antenna array configurations.
Includes functions to evolve the population, save/load configurations, and manage evolution progress.
"""

import os
import json
import time
import platform
from typing import Tuple, Optional
import torch
from IPython.display import clear_output
from config import device, report_evolutions, save_evolutions
from util import clear_cache
from generate import evolve_population
from reward import compute_reward_vector

# Define type for population tuple
Population = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

def evolve(population: Population, device: str = device, max_evolutions: int = 100000, save_dir: str = "saved_populations") -> Population:
    """
    Evolve a population of antenna configurations to maximize the reward function.

    Args:
        population (Population): Tuple of (amplitude_mats, phase_mats, position_mats, power_mats, shape_mats).
        device (str): Device for tensor operations ('cpu', 'cuda', or 'mps').
        max_evolutions (int): Maximum number of evolution iterations.
        save_dir (str): Directory to save population snapshots.

    Returns:
        Population: Best configuration found during evolution.
    """
    os.makedirs(save_dir, exist_ok=True)
    amplitude_mats, phase_mats, position_mats, power_mats, shape_mats = population

    best_reward: torch.Tensor = torch.tensor(-20.0, device=device)
    best_config: Optional[Population] = None

    for evolution in range(max_evolutions):
        clear_cache(device)
        start_time: float = time.time()
        reward_vector: torch.Tensor = compute_reward_vector(population, device=device)

        current_best_index: int = int(torch.argmax(reward_vector).item())
        current_best_reward: torch.Tensor = reward_vector[current_best_index]

        if current_best_reward >= best_reward:
            best_reward = current_best_reward
            best_config = tuple(tensor[current_best_index].unsqueeze(0) for tensor in population)

        if (evolution + 1) % report_evolutions == 0:
            avg_reward: float = torch.mean(reward_vector).item()
            time_per_evolution: float = (time.time() - start_time) / report_evolutions
            print(f"Evolution {evolution + 1}:")
            print(f"  Best Reward: {current_best_reward.item():.2f}")
            print(f"  Overall Best Reward: {best_reward.item():.2f}")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Time per Evolution: {time_per_evolution:.2f} seconds")

        if (evolution + 1) % save_evolutions == 0:
            clear_output(wait=True)
            timestamp: int = int(time.time())
            host: str = platform.system()
            save_path: str = os.path.join(save_dir, f"population_{timestamp}_{host}.json")
            try:
                save_population(population, reward_vector, save_path)
                print(f"Population saved to {save_path}")
                print(f"Overall Best Reward: {current_best_reward.item():.2f}")
            except Exception as e:
                print(f"Error saving population: {e}")

        population = evolve_population(population, reward_vector, device=device)

    if best_config is None:
        raise RuntimeError("No best configuration was found during evolution.")
    return best_config


def save_population(population: Population, reward_vector: torch.Tensor, save_path: str) -> None:
    """
    Save the current population and corresponding rewards to a JSON file.

    Args:
        population (Population): (amplitude_mats, phase_mats, position_mats, power_mats, shape_mats).
        reward_vector (torch.Tensor): Reward for each configuration.
        save_path (str): File path for saving the population.
    """
    amplitude_mats, phase_mats, position_mats, power_mats, shape_mats = population

    population_data = {
        "amplitude_mats": amplitude_mats.cpu().tolist(),
        "phase_mats": phase_mats.cpu().tolist(),
        "position_mats": position_mats.cpu().tolist(),
        "power_mats": power_mats.cpu().tolist(),
        "shape_mats": shape_mats.cpu().tolist(),
        "reward_vector": reward_vector.tolist()
    }

    try:
        with open(save_path, "w") as f:
            json.dump(population_data, f)
    except Exception as e:
        raise IOError(f"Failed to save population to {save_path}: {e}") from e


def load_population(save_path: str, device: str = device) -> Tuple[Population, torch.Tensor]:
    """
    Load a population configuration from a JSON file.

    Args:
        save_path (str): Path to the JSON file.
        device (str): Device for tensor operations ('cpu', 'cuda', or 'mps').

    Returns:
        Tuple[Population, torch.Tensor]: (population, reward_vector)
    """
    try:
        with open(save_path, "r") as f:
            population_data = json.load(f)
    except Exception as e:
        raise IOError(f"Failed to load population from {save_path}: {e}") from e

    amplitude_mats = torch.tensor(population_data["amplitude_mats"], dtype=torch.float32, device=device)
    phase_mats = torch.tensor(population_data["phase_mats"], dtype=torch.float32, device=device)
    position_mats = torch.tensor(population_data["position_mats"], dtype=torch.float32, device=device)
    power_mats = torch.tensor(population_data["power_mats"], dtype=torch.float32, device=device)
    shape_mats = torch.tensor(population_data["shape_mats"], dtype=torch.int8, device=device)
    reward_vector = torch.tensor(population_data["reward_vector"], dtype=torch.float32, device=device)

    return (amplitude_mats, phase_mats, position_mats, power_mats, shape_mats), reward_vector


def get_latest_population_file(save_dir: str = "saved_populations") -> Optional[str]:
    """
    Retrieve the most recent population file from the specified directory.

    Args:
        save_dir (str): Directory containing saved population files.

    Returns:
        Optional[str]: Path to the latest saved population file, or None if not found.
    """
    if not os.path.exists(save_dir) or not os.listdir(save_dir):
        print("⚠️ No saved population files found.")
        return None

    population_files = [f for f in os.listdir(save_dir) if f.startswith("population_") and f.endswith(".json")]
    if not population_files:
        print("⚠️ No valid population files found.")
        return None

    population_files.sort(reverse=True)
    latest_file: str = os.path.join(save_dir, population_files[0])
    return latest_file


def load_latest_population(device: str = device, save_dir: str = "saved_populations") -> Optional[Tuple[Population, torch.Tensor]]:
    """
    Load the most recent saved population.

    Args:
        device (str): Device for tensor operations ('cpu', 'cuda', or 'mps').
        save_dir (str): Directory containing saved population files.

    Returns:
        Optional[Tuple[Population, torch.Tensor]]: (population, reward_vector) if a saved file exists, otherwise None.
    """
    latest_file = get_latest_population_file(save_dir)
    if latest_file is None:
        return None
    return load_population(latest_file, device=device)
