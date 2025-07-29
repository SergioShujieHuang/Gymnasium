import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(tuple(transition))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*samples))

    def __len__(self):
        return len(self.buffer)

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return q_net(state_tensor).argmax().item()

# Hyperparameters
gamma = 0.99
lr = 1e-3
batch_size = 64
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 500
target_update_freq = 10

env = gym.make("CartPole-v1", render_mode=None)
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

q_net = QNetwork(obs_size, n_actions)
target_net = QNetwork(obs_size, n_actions)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=lr)
replay_buffer = ReplayBuffer()

epsilon = epsilon_start
steps_done = 0

def shaped_reward(state, reward, done):
    x, x_dot, theta, theta_dot = state
    position_penalty = abs(x) * 1.5
    angle_penalty = abs(theta) * 3.0
    shaped = reward - (position_penalty + angle_penalty)
    return shaped if not done else -50

num_episodes = 300
episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    while True:
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)

        shaped_r = shaped_reward(next_state, reward, terminated or truncated)
        replay_buffer.push(state, action, shaped_r, next_state, terminated or truncated)
        state = next_state
        total_reward += reward
        steps_done += 1

        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            q_values = q_net(states).gather(1, actions)
            next_actions = q_net(next_states).argmax(1, keepdim=True)
            next_q_values = target_net(next_states).gather(1, next_actions)
            expected_q_values = rewards + (1 - dones) * gamma * next_q_values

            loss = nn.MSELoss()(q_values, expected_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)

    # Update target network
    if episode % target_update_freq == 0:
        target_net.load_state_dict(q_net.state_dict())

    if episode % 10 == 0:
        print(f"Episode {episode}, reward: {total_reward:.1f}, epsilon: {epsilon:.3f}")
        test_env = gym.make("CartPole-v1", render_mode="human")
        test_state, _ = test_env.reset()
        while True:
            with torch.no_grad():
                action = q_net(torch.tensor(test_state, dtype=torch.float32).unsqueeze(0)).argmax().item()
            test_state, _, done, _, _ = test_env.step(action)
            if done:
                break
        test_env.close()
