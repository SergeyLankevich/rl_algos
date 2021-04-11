import os
import gym
import datetime
import numpy as np
from tensorflow import keras
from Memory import Memory, ETDMemory, RingMemory
from Policies import Policy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from Base import Agent, MemoryAgent, PlayingData

# FILE CONTAINS DQN, Policy Grad & A2C Agents
# TODO: Write up Proximal Policy Optimization agent (like A2C but avoids taking large gradient steps)
# TODO: finish DQN-agent (epochs types issues)


class DQNAgent(MemoryAgent):

    @staticmethod
    def get_dueling_output_layer(n_actions, dueling_type='avg'):
        assert dueling_type in ['avg', 'max', 'naive']

        def layer(x1, x2):
            x1 = keras.layers.Dense(1)(x1)
            x2 = keras.layers.Dense(n_actions)(x2)
            x = keras.layers.Concatenate()([x1, x2])
            if dueling_type == 'avg':
                def dueling(a):
                    return (K.expand_dims(a[:, 0], -1) + a[:, 1:] -
                            K.mean(a[:, 1:], axis=1, keepdims=True))
            elif dueling_type == 'max':
                def dueling(a):
                    return (K.expand_dims(a[:, 0], -1) + a[:, 1:] -
                            K.max(a[:, 1:], axis=1, keepdims=True))
            else:
                def dueling(a):
                    return K.expand_dims(a[:, 0], -1) + a[:, 1:]
            return keras.layers.Lambda(dueling, output_shape=(n_actions,),
                                       name='q_output')(x)

        return layer

    def __init__(self, policy, q_model, discounted_rate,
                 create_memory=lambda shape, dtype: Memory(),
                 enable_target=True, enable_double=False,
                 enable_per=False):

        MemoryAgent.__init__(self, q_model.output_shape[1], policy)
        self.q_model = q_model
        self.q_model.compiled_loss.build(
            tf.zeros(self.q_model.output_shape[1:])
        )
        self.target_q_model = None
        self.enable_target = enable_target or enable_double
        self.enable_double = enable_double
        if self.enable_target:
            self.target_q_model = keras.models.clone_model(q_model)
            self.target_q_model.compile(optimizer='sgd', loss='mse')
        else:
            self.target_q_model = self.q_model
        self.discounted_rate = discounted_rate
        self.states = create_memory(self.q_model.input_shape,
                                    keras.backend.floatx())
        self.next_states = create_memory(self.q_model.input_shape,
                                         keras.backend.floatx())
        self.actions = create_memory(self.q_model.output_shape,
                                     keras.backend.floatx())
        self.rewards = create_memory((None,),
                                     keras.backend.floatx())
        self.terminals = create_memory((None,),
                                       keras.backend.floatx())
        self.memory = {
            'states': self.states, 'next_states': self.next_states,
            'actions': self.actions, 'rewards': self.rewards,
            'terminals': self.terminals
        }
        if isinstance(self.memory['states'], ETDMemory):
            self.time_distributed_states = np.array([
                self.memory['states'].buffer[0]
                for _ in range(self.memory['states'].num_time_steps)
            ])
        if enable_per:
            self.per_losses = create_memory((None,),
                                            keras.backend.floatx())
            self.memory['per_losses'] = self.per_losses
            self.max_loss = 100.0
        else:
            self.per_losses = None
        self.action_identity = np.identity(self.n_actions)
        self.total_steps = 0
        self.metric = keras.metrics.Mean(name='loss')

        self._tf_train_step = tf.function(
            self._train_step,
            input_signature=(tf.TensorSpec(shape=self.q_model.input_shape,
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=self.q_model.input_shape,
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None, None),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=keras.backend.floatx()))
        )

    def select_action(self, state, training=False):

        if (self.time_distributed_states is not None
                and state.shape == self.q_model.input_shape[2:]):
            self.time_distributed_states = np.roll(
                self.time_distributed_states, -1
            )
            self.time_distributed_states[-1] = state
            state = self.time_distributed_states

        def _select_action():
            q_values = self.q_model(np.expand_dims(state, axis=0), training=False)[0].numpy()
            return q_values

        return self.policy.select_action(_select_action,
                                         training=training)

    def set_playing_data(self, training=False, memorizing=False,
                         learns_in_episode=False, batch_size=None,
                         mini_batch=0, epochs=1, repeat=1,
                         target_update_interval=1, tau=1.0, verbose=True):

        learning_params = {'batch_size': batch_size,
                           'mini_batch': mini_batch,
                           'epochs': epochs,
                           'repeat': repeat,
                           'target_update_interval': target_update_interval,
                           'tau': tau,
                           'verbose': verbose}
        self.playing_data = PlayingData(training, memorizing, epochs,
                                        learns_in_episode, learning_params)

    def add_memory(self, state, action, new_state, reward, terminal):

        self.states.add(np.array(state))
        self.next_states.add(np.array(new_state))
        self.actions.add(self.action_identity[action])
        self.rewards.add(reward)
        self.terminals.add(0 if terminal else 1)
        if self.per_losses is not None:
            self.per_losses.add(self.max_loss)

    def update_target(self, tau):
        if tau == 1.0:
            self.target_q_model.set_weights(self.q_model.get_weights())
        else:
            tws = self.target_q_model.trainable_variables
            ws = self.q_model.trainable_variables
            for ndx in range(len(tws)):
                tws[ndx] = ws[ndx] * tau + tws[ndx] * (1 - tau)

    def _train_step(self, states, next_states,
                    actions, terminals, rewards):

        if self.enable_double:
            q_values = self.q_model(next_states, training=False)
            actions = tf.argmax(q_values, axis=-1)
            q_values = self.target_q_model(next_states, training=False)
            q_values = tf.squeeze(tf.gather(q_values, actions[:, tf.newaxis], axis=-1, batch_dims=1))
            actions = tf.one_hot(actions, self.n_actions,
                                 dtype=q_values.dtype)
        else:
            q_values = self.target_q_model(next_states, training=False)
            q_values = tf.reduce_max(q_values, axis=-1)
        q_values = (rewards + self.discounted_rate * q_values * terminals)

        with tf.GradientTape() as tape:
            y_pred = self.q_model(states, training=True)
            if len(self.q_model.losses) > 0:
                reg_loss = tf.math.add_n(self.q_model.losses)
            else:
                reg_loss = 0
            y_true = (y_pred * (1 - actions) +
                      q_values[:, tf.newaxis] * actions)
            loss = self.q_model.compiled_loss._losses[0].fn(y_true, y_pred) + reg_loss
        grads = tape.gradient(loss, self.q_model.trainable_variables)
        self.q_model.optimizer.apply_gradients(
            zip(grads, self.q_model.trainable_variables)
        )
        self.metric(loss)

        return tf.reduce_sum(tf.abs(y_true - y_pred), axis=-1)

    def _train(self, states, next_states, actions, terminals,
               rewards, epochs, batch_size, verbose=True):

        length = states.shape[0]
        float_type = keras.backend.floatx()
        batches = tf.data.Dataset.from_tensor_slices(
            (states.astype(float_type),
             next_states.astype(float_type),
             actions.astype(float_type),
             terminals.astype(float_type),
             rewards.astype(float_type))
        ).batch(batch_size)
        losses = []
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f'Epoch {epoch}/{epochs}')

            count = 0
            for batch in batches:
                if epoch == epochs:
                    losses.append(self._tf_train_step(*batch).numpy())
                else:
                    self._tf_train_step(*batch)
                count += np.minimum(batch_size, length - count)
                if verbose:
                    print(f'{count}/{length}', end='\r')
            loss_results = self.metric.result()
            self.metric.reset_states()
            if verbose:
                print(f'{count}/{length} - '
                      f'loss: {loss_results}')
        return np.hstack(losses)

    def learn(self, batch_size=None, mini_batch=0,
              epochs=1, repeat=1,
              target_update_interval=1, tau=1.0, verbose=True):

        self.total_steps += 1
        if batch_size is None:
            batch_size = len(self.states)
        if 0 < mini_batch < len(self.states):
            length = mini_batch
        else:
            length = len(self.states)

        for count in range(1, repeat + 1):
            if verbose:
                print(f'Repeat {count}/{repeat}')

            if self.per_losses is None:
                arrays, indexes = self.states.shuffled_sample(
                    [self.states, self.next_states, self.actions,
                     self.terminals, self.rewards],
                    length
                )
            else:
                per_losses = self.per_losses.array()
                self.max_loss = per_losses.max()
                per_losses = per_losses / per_losses.sum()
                arrays, indexes = self.states.shuffled_sample(
                    [self.states, self.next_states, self.actions,
                     self.terminals, self.rewards],
                    length, weights=per_losses
                )
            losses = self._train(*arrays, epochs, batch_size, verbose=verbose)
            if self.per_losses is not None:
                for ndx, loss in zip(indexes, losses):
                    self.per_losses[ndx] = loss

            if (self.enable_target
                    and self.total_steps % target_update_interval == 0):
                self.update_target(tau)

    def load(self, path, load_model=True, load_data=True):
        note = MemoryAgent.load(self, path, load_data=load_data)
        if load_model:
            with open(os.path.join(path, 'q_model.json'), 'r') as file:
                self.q_model = tf.keras.models.model_from_json(file.read())
            self.q_model.load_weights(os.path.join(path, 'q_weights.h5'))
            if self.enable_target:
                self.target_q_model = keras.models.clone_model(self.q_model)
                self.target_q_model.compile(optimizer='sgd', loss='mse')
            else:
                self.target_q_model = self.q_model
        return note

    def save(self, path, save_model=True, save_data=True, note='DQNAgent Save'):
        path = MemoryAgent.save(self, path, save_data=save_data, note=note)
        if save_model:
            with open(os.path.join(path, 'q_model.json'), 'w') as file:
                file.write(self.q_model.to_json())
            self.q_model.save_weights(os.path.join(path, 'q_weights.h5'))
        return path


class PGAgent(MemoryAgent):

    def __init__(self, a_model, discounted_rate,
                 create_memory=lambda shape, dtype: Memory()):

        output_shape = a_model.output_shape
        if isinstance(output_shape, list):
            output_shape = output_shape[-1]
        MemoryAgent.__init__(self, output_shape[1], Policy())
        self.a_model = a_model
        self.discounted_rate = discounted_rate
        self.states = create_memory(self.a_model.input_shape,
                                    keras.backend.floatx())
        self.actions = create_memory(output_shape,
                                     keras.backend.floatx())
        self.d_rewards = create_memory((None,),
                                       keras.backend.floatx())
        self.memory = {
            'states': self.states, 'actions': self.actions,
            'd_rewards': self.d_rewards,
        }
        if isinstance(self.memory['states'], ETDMemory):
            self.time_distributed_states = np.array([
                self.memory['states'].buffer[0]
                for _ in range(self.memory['states'].num_time_steps)
            ])
        self.episode_rewards = []
        self.action_identity = np.identity(self.n_actions)
        self.metric = keras.metrics.Mean(name='loss')
        self._tf_train_step = tf.function(
            self._train_step,
            input_signature=(tf.TensorSpec(shape=self.a_model.input_shape,
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None, None),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(),
                                           dtype=keras.backend.floatx()))
        )

    def select_action(self, state, training=False):
        if (self.time_distributed_states is not None
                and state.shape == self.a_model.input_shape[2:]):
            self.time_distributed_states = np.roll(
                self.time_distributed_states, -1
            )
            self.time_distributed_states[-1] = state
            state = self.time_distributed_states

        actions = self.a_model(np.expand_dims(state, axis=0), training=False)
        if isinstance(actions, list):
            return actions[-1][0].numpy()
        return np.random.choice(np.arange(self.n_actions),
                                p=actions[0].numpy())

    def set_playing_data(self, training=False, memorizing=False,
                         batch_size=None, mini_batch=0, epochs=1,
                         repeat=1, entropy_coef=0, verbose=True):

        learning_params = {'batch_size': batch_size,
                           'mini_batch': mini_batch,
                           'epochs': epochs,
                           'repeat': repeat,
                           'entropy_coef': entropy_coef,
                           'verbose': verbose}
        self.playing_data = PlayingData(training, memorizing, epochs,
                                        False, learning_params)

    def add_memory(self, state, action, new_state, reward, terminal):

        self.states.add(np.array(state))
        self.actions.add(self.action_identity[action])
        self.episode_rewards.append(reward)

    def end_episode(self):
        if len(self.episode_rewards) > 0:
            d_reward = 0
            d_reward_list = []
            for reward in reversed(self.episode_rewards):
                d_reward *= self.discounted_rate
                d_reward += reward
                d_reward_list.append(d_reward)
            self.episode_rewards.clear()
            for d_reward in reversed(d_reward_list):
                self.d_rewards.add(d_reward)

        MemoryAgent.end_episode(self)

    def _train_step(self, states, d_rewards, actions, entropy_coef):

        with tf.GradientTape() as tape:
            y_pred = self.a_model(states, training=True)
            log_y_pred = tf.math.log(y_pred + keras.backend.epsilon())
            log_probs = tf.reduce_sum(
                actions * log_y_pred, axis=1
            )
            loss = -tf.reduce_mean(d_rewards * log_probs)
            entropy = tf.reduce_sum(
                y_pred * log_y_pred, axis=1
            )
            loss += tf.reduce_mean(entropy) * entropy_coef
        grads = tape.gradient(loss, self.a_model.trainable_variables)
        self.a_model.optimizer.apply_gradients(
            zip(grads, self.a_model.trainable_variables)
        )
        self.metric(loss)

    def _train(self, states, d_rewards, actions,
               epochs, batch_size, entropy_coef, verbose=True):

        length = states.shape[0]
        float_type = keras.backend.floatx()
        batches = tf.data.Dataset.from_tensor_slices(
            (states.astype(float_type),
             d_rewards.astype(float_type),
             actions.astype(float_type))
        ).batch(batch_size)
        entropy_coef = tf.constant(entropy_coef,
                                   dtype=float_type)
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f'Epoch {epoch}/{epochs}')
            count = 0
            for batch in batches:
                self._tf_train_step(*batch, entropy_coef)
                count += np.minimum(batch_size, length - count)
                if verbose:
                    print(f'{count}/{length}', end='\r')
            loss_results = self.metric.result()
            self.metric.reset_states()
            if verbose:
                print(f'{count}/{length} - '
                      f'loss: {loss_results}')
        return loss_results

    def learn(self, batch_size=None, mini_batch=0,
              epochs=1, repeat=1, entropy_coef=0, verbose=True):

        if batch_size is None:
            batch_size = len(self.states)
        if 0 < mini_batch < len(self.states):
            length = mini_batch
        else:
            length = len(self.states)

        for count in range(1, repeat + 1):
            if verbose:
                print(f'Repeat {count}/{repeat}')

            arrays, _ = self.states.shuffled_sample(
                [self.states, self.d_rewards, self.actions], length
            )
            std = arrays[1].std()
            if std == 0:
                return False
            arrays[1] = (arrays[1] - arrays[1].mean()) / std
            self._train(*arrays, epochs, batch_size,
                        entropy_coef, verbose=verbose)

    def load(self, path, load_model=True, load_data=True):
        note = MemoryAgent.load(self, path, load_data=load_data)
        if load_model:
            with open(os.path.join(path, 'a_model.json'), 'r') as file:
                self.a_model = tf.keras.models.model_from_json(file.read())
            self.a_model.load_weights(os.path.join(path, 'a_weights.h5'))
        return note

    def save(self, path, save_model=True, save_data=True, note='PGAgent Save'):

        path = MemoryAgent.save(self, path, save_data=save_data, note=note)
        if save_model:
            with open(os.path.join(path, 'a_model.json'), 'w') as file:
                file.write(self.a_model.to_json())
            self.a_model.save_weights(os.path.join(path, 'a_weights.h5'))
        return path


class A2CAgent(PGAgent):

    def __init__(self, a_model, c_model, discounted_rate,
                 lambda_rate=0, create_memory=lambda shape, dtype: Memory()):
        PGAgent.__init__(self, a_model, discounted_rate,
                         create_memory=create_memory)
        self.c_model = c_model
        self.c_model.compiled_loss.build(
            tf.zeros(self.c_model.output_shape[1:])
        )
        self.lambda_rate = lambda_rate
        if lambda_rate != 0:
            self.terminals = create_memory((None,),
                                           keras.backend.floatx())
            self.rewards = create_memory((None,),
                                         keras.backend.floatx())
            self.memory['terminals'] = self.terminals
            self.memory['rewards'] = self.rewards
        self.metric_c = keras.metrics.Mean(name='critic_loss')
        self._tf_train_step = tf.function(
            self._train_step,
            input_signature=(tf.TensorSpec(shape=self.a_model.input_shape,
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None,),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(None, None),
                                           dtype=keras.backend.floatx()),
                             tf.TensorSpec(shape=(),
                                           dtype=keras.backend.floatx()))
        )

    def add_memory(self, state, action, new_state, reward, terminal):
        self.states.add(np.array(state))
        self.actions.add(self.action_identity[action])
        self.episode_rewards.append(reward)
        if self.lambda_rate > 0:
            self.terminals.add(terminal)
            self.rewards.add(reward)

    def end_episode(self):
        if len(self.episode_rewards) > 0:
            d_reward = 0
            d_reward_list = []
            for reward in reversed(self.episode_rewards):
                d_reward *= self.discounted_rate
                d_reward += reward
                d_reward_list.append(d_reward)
            self.episode_rewards.clear()
            for d_reward in reversed(d_reward_list):
                self.d_rewards.add(d_reward)
            if self.lambda_rate > 0:
                self.terminals[-1] = True

        MemoryAgent.end_episode(self)

    def _train_step(self, states, d_rewards, advantages,
                    actions, entropy_coef):

        with tf.GradientTape() as tape:
            value_pred = tf.squeeze(self.c_model(states, training=True))
            if len(self.c_model.losses) > 0:
                reg_loss = tf.math.add_n(self.c_model.losses)
            else:
                reg_loss = 0
            loss = self.c_model.compiled_loss._losses[0].fn(d_rewards, value_pred)
            loss = loss + reg_loss
        grads = tape.gradient(loss, self.c_model.trainable_variables)
        self.c_model.optimizer.apply_gradients(
            zip(grads, self.c_model.trainable_variables)
        )
        self.metric_c(loss)

        with tf.GradientTape() as tape:
            action_pred = self.a_model(states, training=True)
            log_action_pred = tf.math.log(
                action_pred + keras.backend.epsilon()
            )
            log_probs = tf.reduce_sum(
                actions * log_action_pred, axis=1
            )
            if self.lambda_rate == 0:
                advantages = (d_rewards - value_pred)
            loss = -tf.reduce_mean(advantages * log_probs)
            entropy = tf.reduce_sum(action_pred * log_action_pred, axis=1)
            loss += tf.reduce_mean(entropy) * entropy_coef

        grads = tape.gradient(loss, self.a_model.trainable_variables)
        self.a_model.optimizer.apply_gradients(
            zip(grads, self.a_model.trainable_variables)
        )
        self.metric(loss)

    def _train(self, states, d_rewards, advantages, actions, epochs, batch_size, entropy_coef, verbose=True):
        length = states.shape[0]
        float_type = keras.backend.floatx()
        batches = tf.data.Dataset.from_tensor_slices(
            (states.astype(float_type),
             d_rewards.astype(float_type),
             advantages.astype(float_type),
             actions.astype(float_type))
        ).batch(batch_size)
        entropy_coef = tf.constant(entropy_coef,
                                   dtype=float_type)
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f'Epoch {epoch}/{epochs}')
            count = 0
            for batch in batches:
                self._tf_train_step(*batch, entropy_coef)
                count += np.minimum(batch_size, length - count)
                if verbose:
                    print(f'{count}/{length}', end='\r')
            actor_loss_results = self.metric.result()
            critic_loss_results = self.metric_c.result()
            self.metric.reset_states()
            self.metric_c.reset_states()
            if verbose:
                print(f'{count}/{length} - '
                      f'actor_loss: {actor_loss_results} - '
                      f'critic_loss: {critic_loss_results}')
        return critic_loss_results

    def learn(self, batch_size=None, mini_batch=0,
              epochs=1, repeat=1, entropy_coef=0, verbose=True):
        if batch_size is None:
            batch_size = len(self.states)
        if 0 < mini_batch < len(self.states):
            length = mini_batch
        else:
            length = len(self.states)

        for count in range(1, repeat + 1):
            if verbose:
                print(f'Repeat {count}/{repeat}')

            if self.lambda_rate == 0:
                advantages_arr = np.empty(length)
            else:
                # c_model predict on batches if large?
                values = tf.squeeze(
                    self.c_model(self.states.array())
                ).numpy()
                advantages = np.empty(len(self.rewards))
                for ndx in reversed(range(len(self.rewards))):
                    delta = self.rewards[ndx] - values[ndx]
                    if not self.terminals[ndx]:
                        delta += self.discounted_rate * values[ndx + 1]
                    if self.terminals[ndx]:
                        advantage = 0
                    advantage = (delta + self.discounted_rate * self.lambda_rate * advantage)
                    advantages[ndx] = advantage

            arrays, indexes = self.states.shuffled_sample(
                [self.states, self.d_rewards, self.actions], length
            )

            if self.lambda_rate == 0:
                arrays = [arrays[0], arrays[1], advantages_arr, arrays[2]]
                std = arrays[1].std()
                if std == 0:
                    return False
                arrays[1] = (arrays[1] - arrays[1].mean()) / std
            else:
                arrays = [arrays[0], arrays[1], advantages[indexes], arrays[2]]
                std = arrays[2].std()
                if std == 0:
                    return False
                arrays[2] = (arrays[2] - arrays[2].mean()) / std

            self._train(*arrays, epochs, batch_size,
                        entropy_coef, verbose=verbose)

    def load(self, path, load_model=True, load_data=True, custom_objects=None):
        note = PGAgent.load(
            self, path, load_model=load_model, load_data=load_data)
        if load_model:
            with open(os.path.join(path, 'c_model.json'), 'r') as file:
                self.c_model = tf.keras.models.model_from_json(
                    file.read(), custom_objects=custom_objects
                )
            self.c_model.load_weights(os.path.join(path, 'c_weights.h5'))
        return note

    def save(self, path, save_model=True,
             save_data=True, note='A2CAgent Save'):
        path = PGAgent.save(self, path, save_model=save_model,
                            save_data=save_data, note=note)
        if save_model:
            with open(os.path.join(path, 'c_model.json'), 'w') as file:
                file.write(self.c_model.to_json())
            self.c_model.save_weights(os.path.join(path, 'c_weights.h5'))
        return path
