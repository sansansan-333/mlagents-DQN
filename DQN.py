from typing import List, Tuple
from keras import models, layers
import tensorflow as tf
import numpy as np
import random
import datetime
import os
from statistics import mean

from functionTimeMeasure import FunctionTimeMeasure

ftm = FunctionTimeMeasure()

class TimeStep: 
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.state = np.zeros(state_size)
        self.action = np.zeros(action_size)
        self.reward = 0
        self.done = 0 # 0 for continue, 1 for terminal
        self.next_state = np.zeros(state_size)
        self.next_action = np.zeros(action_size)

    def __str__(self) -> str:
        return f'state: {self.state}, action: {self.action}, reward: {self.reward}, done: {"True" if self.done == 1 else "False" if self.done == 0 else "Unknown"}, next_state: {self.next_state}, next_action: {self.next_action}'

class ReplayBuffer:
    def __init__(self, size: int):
        self.size = size
        self.buffer = []
        self._p = 0
    
    def append(self, time_step: TimeStep):
        if self.size == 0:
            return

        if len(self.buffer) < self.size:
            self.buffer.append(time_step)
        else:
            self.buffer[self._p] = time_step

        self._p += 1
        self._p = self._p % self.size
        
    def sample(self, batch_size) -> List[TimeStep]:
        ftm.start('sample')
        sample = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        ftm.end('sample')
        return sample

class QNetwork:
    def __init__(self, input_shape: Tuple, output_size: int):
        self.input_shape = input_shape
        self.output_size = output_size
        self.model = models.Sequential([
            layers.Dense(units=16, activation='relu', input_shape=input_shape),
            layers.Dense(units=8, activation='relu'),
            layers.Dense(units=output_size, activation='linear')
        ])

    def compile(self, loss):
        self.model.compile(optimizer='adam', loss=loss)

class DQN:
    def __init__(
        self,
        epsilon,
        gamma,
        start_steps,
        update_interval,
        target_update_interval,
        delta_output_range,

    ):
        self.epsilon = epsilon
        self.gamma = gamma
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.delta_output_range = delta_output_range

        self._step = 0 # step needs to be counted manually by calling count_step()
        self.q_network: QNetwork = None
        self.target_q_network: QNetwork = None


        # tensorboard
        dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        self.train_writer = tf.summary.create_file_writer(dir)


    def is_random(self) -> bool:
        return self._step < self.start_steps or np.random.rand() < self.epsilon

    def is_update(self) -> bool:
        return self._step >= self.start_steps and self._step % self.update_interval == 0

    def is_update_target(self) -> bool:
        return self._step >= self.start_steps and self._step % self.target_update_interval == 0

    def count_step(self):
        self._step += 1

    def select_action(self, state: np.ndarray) -> np.ndarray:
        q_values: np.ndarray = self.q_network.model.predict_on_batch(np.array([state]))[0] # predict_on_batch() is much faster than predict() if executed on single batch 

        print(q_values)
        q_max = np.max(q_values)
        if q_values[0] == q_max:
            action = [0 for _ in range(self.q_network.output_size)]
            action[0] = 1
            return np.asarray(action)
        else: 
            action = [1 if q >= q_max - self.delta_output_range else 0  for q in q_values]
            action[0] = 0
            return np.asarray(action)
    
    def get_random_action(self) -> np.ndarray:
        size = self.q_network.output_size
        rand = np.random.rand(size)

        rand_max = np.max(rand)
        if rand[0] == rand_max:
            action = [0 for _ in range(size)]
            action[0] = 1
            return np.asarray(action)
        else:
            action = [1 if r >= rand_max - self.delta_output_range else 0   for r in rand]
            action[0] = 0
            return np.asarray(action)


    def update(self, batch: List[TimeStep]):
        x_state = np.zeros((len(batch), *self.q_network.input_shape))
        y_target = np.zeros((len(batch), self.q_network.output_size))
        next_qs = self.target_q_network.model.predict(np.array([time_step.next_state for time_step in batch]))
        for i in range(len(batch)):
            x_state[i] = batch[i].state
            y_target[i] = batch[i].reward + (1 - batch[i].done) * self.gamma * np.max(next_qs[i])

        history = self.q_network.model.fit(
            x=x_state,
            y=y_target,
            batch_size=1,
            epochs=1,
            verbose=1
        )

    def update_target(self):
        self.target_q_network.model.set_weights(self.q_network.model.get_weights())

    def set_q_networks(self, input_shape: Tuple, output_size: int):
        self.q_network = QNetwork(input_shape, output_size)
        self.target_q_network = QNetwork(input_shape, output_size)
        self.q_network.compile(loss=self._get_custom_loss())
        self.target_q_network.compile(loss=self._get_custom_loss())

    def _get_custom_loss(self):
        def loss(y_target, y_current_q):
            l = 0.5 * tf.math.pow(y_target - y_current_q, 2)
            return l
        return loss