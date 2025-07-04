import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())
