from typing import List, Tuple
from keras import models, layers
import tensorflow as tf
import numpy as np
import random
import datetime
import sys
import os
import glob
import json
from statistics import mean

from functionTimeMeasure import FunctionTimeMeasure

ftm = FunctionTimeMeasure()

class TimeStep: 
    def __init__(self, state_size: int, action_size: int):
        self.state = np.zeros(state_size)
        self.action = np.zeros(action_size)
        self.reward: float = 0
        self.done: int = 0 # 0 for continue, 1 for terminal
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
        sample = random.sample(self.buffer, min(batch_size, len(self.buffer)))
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
        epsilon: float,
        gamma: float,
        start_steps: int,
        update_interval: int,
        target_update_interval: int,
        demonstrations_path: str = None
    ):
        self.epsilon = epsilon
        self.gamma = gamma
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval

        self._step = 0 # step needs to be counted manually by calling count_step()
        self.q_network: QNetwork = None
        self.target_q_network: QNetwork = None

        # tensorboard
        dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        self.train_writer = tf.summary.create_file_writer(dir)

        # reward shaping
        if demonstrations_path == None:
            self.reward_shaping = False
        else:
            self.demonstrations: List[TimeStep] = self._load_demonstrations(demonstrations_path)
            self.reward_shaping = len(self.demonstrations) > 0


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

        action = np.zeros((self.q_network.output_size))
        action[q_values.argmax()] = 1
        return action
        
    
    def get_random_action(self) -> np.ndarray:
        size = self.q_network.output_size
        rand = np.random.rand(size)

        action = np.zeros((self.q_network.output_size))
        action[rand.argmax()] = 1
        return action


    def update(self, batch: List[TimeStep]):
        ftm.start("update 1")
        x_state = np.zeros((len(batch), *self.q_network.input_shape))
        y_target = np.zeros((len(batch), self.q_network.output_size))
        next_qs = self.target_q_network.model.predict(np.array([time_step.next_state for time_step in batch]))
        current_qs = self.q_network.model.predict(np.array([time_step.next_state for time_step in batch]))
        for i in range(len(batch)):
            x_state[i] = batch[i].state
            if self.reward_shaping:
                additional_reward = self._calc_additional_reward(batch[i])
            else:
                additional_reward = 0

            # モデルを更新するとき、その時間ステップで行った行動a_tに対応するモデルの出力のみを更新する
            # なので誤差は[0, 0, 0.2, 0, 0, ...] のように一か所だけが０以外になる
            y_target[i] = (batch[i].reward + additional_reward + (1 - batch[i].done) * self.gamma * np.max(next_qs[i])) * batch[i].action    +    current_qs[i] * (1 - batch[i].action)
        ftm.end("update 1")
        print(current_qs[0])
        
        ftm.start("update 2")
        self.q_network.model.fit(
            x=x_state,
            y=y_target,
            batch_size=1,
            epochs=1,
            verbose=1
        )
        ftm.end("update 2")


    def update_target(self):
        self.target_q_network.model.set_weights(self.q_network.model.get_weights())

    def set_q_networks(self, input_shape: Tuple, output_size: int):
        self.q_network = QNetwork(input_shape, output_size)
        self.target_q_network = QNetwork(input_shape, output_size)
        self.q_network.compile(loss=self._get_custom_loss())
        self.target_q_network.compile(loss=self._get_custom_loss())

    def _get_custom_loss(self):
        def loss(y_target: tf.Tensor, y_current_q: tf.Tensor):
            l = 0.5 * tf.math.pow(y_target - y_current_q, 2)
            return l

        return loss

    def _calc_additional_reward(self, time_step: TimeStep) -> float: 
        f = self.gamma * self._calc_similarity_to_demonstrations(time_step.next_state, time_step.next_action) - self._calc_similarity_to_demonstrations(time_step.state, time_step.action)
        return f

    def _calc_similarity_to_demonstrations(self, state: np.ndarray, action: np.ndarray) -> float:
        if len(self.demonstrations) == 0:
            return 0

        # find the same action, compare state to calculate similarity and return the highest value
        max_sim = -1 # -1 is the minimum value of cosine similarity
        for d in self.demonstrations:
            if np.array_equal(d.action, action):
                # cosine similarity
                sim = np.dot(d.state, state) / (np.linalg.norm(d.state) * np.linalg.norm(state))
                max_sim = max(max_sim, sim)

        return max_sim
    
    def _load_demonstrations(self, path: str):
        '''
        Load demonstrations files and store into a list of TimeStep.
        Note that the focused player is set "player 1". You may need to change it depending on your recordings.
        '''
        demonstrations = []
        focused_player = 1
        state_key = f'p{focused_player}StateVector'
        action_key = f'p{focused_player}ActionVector'

        for file_name in glob.glob(os.path.join(path, '*.json')):
            file = open(file_name)
            demo_json = json.load(file)

            state_size = len(demo_json['frames'][0][state_key])
            action_size = len(demo_json['frames'][0][action_key])
            for i in range(len(demo_json['frames']) - 1):
                # TODO: ignore unnecessary frames
                current_frame = demo_json['frames'][i]
                next_frame = demo_json['frames'][i+1]
                d = TimeStep(state_size, action_size)
                d.state = np.asarray(current_frame[state_key])
                d.action = np.asarray(current_frame[action_key])
                d.next_state = np.asarray(next_frame[state_key])
                d.next_action = np.asarray(next_frame[action_key])
                demonstrations.append(d)
        
        return demonstrations
    
dqn = DQN(
        epsilon=0.05,
        gamma=0.999,
        start_steps=100000,
        update_interval=4,
        target_update_interval=10000,
        demonstrations_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demonstrations')
    )
print("constractor finished")

state = [1, 0, 0, 0, 0, 0, 0, 0, 0.2510966, -1]
action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print(dqn._calc_similarity_to_demonstrations(state, action))