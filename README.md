# Project Report: DQN with Reward Shaping on CartPole-v1

## Overview

I implement a reinforcement learning agent using Deep Q-Network to solve the CartPole-v1 environment from Gymnasium. The agent is trained with experience replay and target network mechanisms, and includes a custom shaped reward function to enhance learning efficiency.

## Environment Setup

To ensure all dependencies are properly installed, **please follow both steps below**:

### Step 1: Create Conda environment and install core dependencies

Run the following commands to create and activate the Conda environment with core packages:

```bash
conda env create -f environment.yml
conda activate cartpole-dqn
```

### Step 2: Install additional dependencies via pip

After activating the Conda environment, run:

```bash
pip install -r requirements.txt
```
