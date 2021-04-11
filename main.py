import gym
import numpy as np
import tensorflow as tf
from Memory import Memory
from Environment import Environment, GymWrapper
from Agents import A2CAgent
from tensorflow import keras
from Policies import Policy
import h5py

g_env = gym.make('MountainCar-v0')
max_steps = g_env._max_episode_steps
print(max_steps)
print(g_env.observation_space, g_env.action_space)

env = GymWrapper(g_env)

inputs = keras.layers.Input(shape=env.state_shape)

# Creating models
x = keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal')(inputs)
x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(x)
x = keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(x)

# Build actor model
actor_outputs = keras.layers.Dense(env.n_actions, activation='softmax')(x)
actor = keras.Model(inputs=inputs, outputs=actor_outputs)
actor.compile(optimizer=keras.optimizers.Adam(.001), loss='mse')
actor.summary()

# Build critic model
critic_outputs = keras.layers.Dense(1)(x)
critic = keras.Model(inputs=inputs, outputs=critic_outputs)
critic.compile(optimizer=keras.optimizers.Adam(.001), loss='mse')
critic.summary()

# Create agent
agent = A2CAgent(a_model=actor,
                 c_model=critic,
                 discounted_rate=.99,
                 lambda_rate=.95,
                 create_memory=lambda shape, dtype: Memory(capacity=20000)
                 )


# Training the agent
def end_episode_callback(episode, reward):
    global agent
    if reward > -150:
        old_playing_data = agent.playing_data
        agent.set_playing_data(training=False, memorizing=False)
        result = env.run_episodes(agent=agent,
                                  num_episodes=10,
                                  max_steps=max_steps,
                                  verbose=False,
                                  episode_verbose=False,
                                  render=False
                                  )
        print(f'Validate results: {result}')
        if result >= -110:
            agent.save(save_dir, note=f'A2C_{episode}_{result}')
        agent.playing_data = old_playing_data
        if result >= -100:  # end early
            return True


agent.set_playing_data(
    training=True, memorizing=True,
    batch_size=16, mini_batch=1024,
    epochs=1, repeat=5,
    entropy_coef=0,
    verbose=False
)

save_dir = 'models'
num_episodes = 500
result = env.run_episodes(
    agent, num_episodes=500, max_steps=max_steps,
    verbose=True, episode_verbose=False,
    render=False,
    end_episode_callback=end_episode_callback
)

agent.save(save_dir, note=f'A2C_{result}')
