import tensorflow as tf
import numpy as np
import gymnasium as gym

from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

# Show Environment
LOCAL = True

model = tf.keras.models.load_model('minigrid_model.h5')

env = gym.make(
    "MiniGrid-Empty-Random-6x6-v0",
    render_mode="human" if LOCAL else "rgb_array",
    highlight=False,
    screen_size=640
)

env = FullyObsWrapper(env)
env = ImgObsWrapper(env)

rewards = []

for episode in range(10):
    obs, _ = env.reset()
    step = 0
    terminated = False
    truncated = False

    while not terminated and not truncated and step < 30:
        if LOCAL:
            env.render()
        # Converting observations to 2D array to feed neural network
        action_probs = model.predict(obs.reshape(-1, *obs.shape), verbose=0)
        action = np.argmax(action_probs)
        obs, reward, terminated, truncated, _ = env.step(action)
        step += 1

    print(f"{episode=} {reward=}")
    rewards.append(reward)

env.close()
print(f"mean reward: {np.mean(rewards)}")
