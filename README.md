# RL-FrozenLake

Final project for the Reinforcement Learning course.

---

## Overview

This project implements Reinforcement Learning algorithms (SARSA and Monte Carlo) to solve the FrozenLake environment using Gymnasium. It investigates the effect of different reward shaping strategies on learning efficiency and policy convergence.

---

## Project Structure

* `src/`: Contains the core implementation.
    * `mc_agent.py`: Implementation of the Monte Carlo agent.
    * `sarsa_agent.py`: Implementation of the SARSA agent.
    * `wrappers.py`: Custom Gymnasium wrappers for reward shaping.
    * `maps.py`: Grid configurations.
    * `experiments.py`: Logic for running training loops and collecting data.
    * `hyperparameter_study.py`: Script for sensitivity analysis (Alpha sweep).
* `main.py`: The entry point to run the main experiment suite and generate comparison plots.
* `results/`: Directory where all plots and logs will be saved.

---

## Requirements

* Python 3.10+
* Libraries: `gymnasium`, `numpy`, `pandas`, `seaborn`, `matplotlib`, `tqdm`
* Full dependencies are listed in `requirements.txt`.

---

## Setup & Installation

It is highly recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/DanielChernov99/RL-FrozenLake.git](https://github.com/DanielChernov99/RL-FrozenLake.git)
    cd RL-FrozenLake
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    if not working try this:
    ```
    py -m pip install -r requirements.txt
---

## How to Run

### 1. Main Experiments (Comparison)
To run the full suite of experiments comparing Baseline, Step-Cost, Potential, and Custom shaping for both MC and SARSA:

```bash
py main.py
