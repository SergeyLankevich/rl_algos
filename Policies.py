import numpy as np


class Policy:

    def __init__(self):
        pass

    def select_action(self, action_func, training):
        return action_func()

    def reset(self):
        pass

    def end_episode(self):
        pass


class GreedyPolicy(Policy):

    def __init__(self):
        super().__init__()

    def select_action(self, action_func, training):
        action = np.argmax(action_func())
        return action


class AsceticPolicy(Policy):

    def __init__(self):
        super().__init__()

    def select_action(self, action_func, training):
        action = np.argmin(action_func())
        return action


class StochasticPolicy(Policy):
    def __init__(self, policy, stochasticity_decay_training,
                 stochasticity_testing, action_size):

        super().__init__()
        self.policy = policy
        self.stochasticity_decay_training = stochasticity_decay_training
        self.stochasticity_testing = stochasticity_testing
        self.action_size = action_size

    def select_action(self, action_func, training):
        if training:
            stochasticity = self.stochasticity_decay_training()
        else:
            stochasticity = self.stochasticity_testing
        if np.random.uniform() < stochasticity:
            return np.random.randint(0, self.action_size)
        else:
            return self.policy.select_action(action_func, training)

    def end_episode(self):
        self.stochasticity_decay_training.step()

    def reset(self):
        self.stochasticity_decay_training.reset()


#  TODO:
class NoisyPolicy(Policy):
    pass