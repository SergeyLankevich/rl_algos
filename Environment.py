import gym
import datetime
from typing import NamedTuple, Any
from Memory import Memory, ETDMemory, RingMemory
from Base import Agent, MemoryAgent, PlayingData
import h5py


class TimeStep(NamedTuple):
    step_type: int
    reward: int
    discount: float
    observation: Any  # Env-dependent


class Environment:

    def __init__(self, state_shape, n_actions):
        if isinstance(state_shape, int):
            self.discrete_state_space = state_shape
            state_shape = 1
        else:
            self.discrete_state_space = None
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.state = None

    def reset(self):
        self.state = None
        return self.state

    def step(self, action):
        self.state = None
        return self.state, 0, False

    def run_episode(self, agent, max_steps, random=False, random_bounds=None, render=False, verbose=True):

        assert isinstance(agent, Agent)
        assert isinstance(agent.playing_data, PlayingData)

        total_reward = 0
        state = self.reset()
        if render:
            self.render()
        for step in range(1, max_steps + 1):
            if random:
                if random_bounds is None:
                    action = np.random.randint(0, self.n_actions)
                else:
                    action = np.random.uniform(*random_bounds, size=self.n_actions)
            else:
                action = agent.select_action(
                    state, training=agent.playing_data.training
                )
            new_state, reward, terminal = self.step(action)

            total_reward += reward

            if agent.playing_data.memorizing:
                agent.add_memory(state, action, new_state, reward, terminal)

            state = new_state

            if verbose:
                print(f'Step: {step} - Reward: {reward} '
                      f'- Action: {action}')
            if render:
                self.render()
            if (agent.playing_data.training
                    and agent.playing_data.learns_in_episode
                    and agent.playing_data.epochs > 0):
                agent.learn(**agent.playing_data.learning_params)
            if terminal:
                break
        agent.end_episode()

        if (agent.playing_data.training
                and not agent.playing_data.learns_in_episode
                and agent.playing_data.epochs > 0):
            agent.learn(**agent.playing_data.learning_params)
        return step, total_reward

    def run_episodes(self, agent,
                     num_episodes,
                     max_steps,
                     random=False,
                     random_bounds=None,
                     render=False,
                     verbose=True,
                     episode_verbose=None,
                     end_episode_callback=None):

        if episode_verbose is None:
            episode_verbose = verbose
        total_rewards = 0
        best_reward = 'Unknown'

        for episode in range(1, num_episodes + 1):
            step, total_reward = self.run_episode(
                agent, max_steps, random=random, random_bounds=random_bounds,
                render=render, verbose=episode_verbose,
            )
            total_rewards += total_reward
            if best_reward == 'Unknown' or total_reward > best_reward:
                best_reward = total_reward
            if verbose:
                str_time = datetime.datetime.now().strftime(r'%H:%M:%S')
                if isinstance(agent, MemoryAgent):
                    mem_len = len(next(iter(agent.memory.values())))
                    mem_str = f' - Memory Size: {mem_len}'
                else:
                    mem_str = ''
                print(f'Time: {str_time} - Episode: {episode} - '
                      f'Steps: {step} - '
                      f'Total Reward: {total_reward} - '
                      f'Best Total Reward: {best_reward} - '
                      f'Average Total Reward: {total_rewards / episode}'
                      f'{mem_str}')
            if end_episode_callback is not None:
                end = end_episode_callback(
                    episode, total_reward
                )
                if end:
                    break
        return total_rewards / episode

    def close(self):
        pass

    def render(self):
        pass


class GymWrapper(Environment):

    def __init__(self, gym_env):
        self.gym_env = gym_env
        if isinstance(self.gym_env.observation_space, gym.spaces.Discrete):
            self.discrete_state_space = self.gym_env.observation_space.n
            self.state_shape = 1
        elif isinstance(self.gym_env.observation_space, gym.spaces.Box):
            self.discrete_state_space = None
            self.state_shape = self.gym_env.observation_space.shape
        else:
            raise NotImplementedError('Only Discrete and Box '
                                      'observation spaces '
                                      'are supported')

        if isinstance(self.gym_env.action_space, gym.spaces.Discrete):
            self.n_actions = self.gym_env.action_space.n
        elif isinstance(self.gym_env.observation_space, gym.spaces.Box):
            if len(self.gym_env.action_space.shape) > 1:
                raise NotImplementedError('Box action spaces with more '
                                          'than one dimension are not '
                                          'implemented yet')
            self.n_actions = self.gym_env.action_space.shape[0]
        else:
            raise NotImplementedError('Only Discrete action '
                                      'spaces are supported')

    def reset(self):
        self.state = self.gym_env.reset()
        if self.discrete_state_space is None:
            return self.state
        else:
            return [self.state]

    def step(self, action):
        self.state, reward, terminal, _ = self.gym_env.step(action)
        if self.discrete_state_space is None:
            return self.state, reward, terminal
        else:
            return [self.state], reward, terminal

    def close(self):
        self.gym_env.close()

    def render(self):
        self.gym_env.render()