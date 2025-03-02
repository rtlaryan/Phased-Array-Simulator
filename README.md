# Satellite Antenna Project

This project implements a genetic algorithm (GA) based optimization framework for designing and optimizing satellite antenna array configurations. The code is designed to generate, simulate, and evolve phased array antenna configurations to maximize a reward function based on signal coverage and power efficiency.

## Project Overview

- **Objective:**  
  Optimize the configuration of a satellite antenna array to achieve desired signal coverage and performance using evolutionary algorithms.

- **Key Features:**
  - **Configuration Generation:**  
    Create random and uniform antenna array configurations with variable shapes, phases, and power levels.
  - **Evolutionary Optimization:**  
    Evolve the population of configurations through selection, mutation, and recombination to maximize a reward function.
  - **Signal Simulation:**  
    Compute the array factor and received signal power at observation points on a target map.
  - **Visualization:**  
    Plot array factor patterns and overlay signal strength on geographic map data.
  - **Coordinate Transformations:**  
    Convert geodetic coordinates to Earth-Centered, Earth-Fixed (ECEF) coordinates and to a satellite-centered spherical coordinate system.

## Code Structure

- **config.py:**  
  Defines global parameters, constants, and loads the target map data. It includes physical constants, array dimensions, evolution parameters, and training configurations.

- **evolve.py:**  
  Contains functions for evolving the population of antenna configurations. It manages the evolution loop, tracks the best configuration, and handles saving/loading populations.

- **generate.py:**  
  Provides functions to generate initial populations (random, uniform, and with various phase tapers), as well as utilities for mutating and combining populations.

- **plot.py:**  
  Includes functions to visualize antenna array performance. It plots the 2D array factor pattern, 1D cuts (azimuth and zenith), and overlays signal strength on the target map.

- **reward.py:**  
  Contains functions to evaluate configuration performance by computing coverage percentages, average power levels, and a reward score.

- **simulate.py:**  
  Implements simulation functions that compute the array factor and received signal power based on the antenna configuration.

- **transform.py:**  
  Provides coordinate transformation utilities to convert between geodetic (LLA) and ECEF coordinates and to a satellite-centered spherical system.

- **util.py:**  
  Contains utility functions for memory reporting, clearing caches, padding matrices, computing path loss, and saving populations to MATLAB files.

## How to Run

1. **Dependencies:**  
   Install the required Python packages:
   pip install torch scipy matplotlib numpy ipython

2. **Data File:**  
Ensure that `GAmap.mat` (a MATLAB file containing target map data) is present in the working directory.

3. **Execution:**  
- Use functions from `generate.py` to initialize a population.
- Run the evolution loop by calling the `evolve` function from `evolve.py`.
- Visualize results using the plotting functions in `plot.py`.
- Evaluate performance with functions in `reward.py` and simulate received power with `simulate.py`.
- Example provided in `GA_Antenna.ipynb`.
## License

This project is open-source and available under the MIT License.

## Acknowledgements

This project was developed as part of an effort to optimize satellite antenna configurations using genetic algorithms and advanced signal processing techniques.
