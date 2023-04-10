from random import random
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from copy import copy
import numpy as np
import tensorflow as tf
import os
import random
import logging
import datetime
import pickle

from DQN import (
    DQN, 
    QNetwork, 
    ReplayBuffer, 
    TimeStep, 
    ftm
)
from DQNProgress import DQNProgress
from statisticsSideChannel import StatisticalSideChannel

# This class is dependent on the UFE environment
class DQNTrainer: 
    def __init__(self, env: UnityEnvironment, dqn: DQN, stats_channel: StatisticalSideChannel, max_steps: int, batch_size: int, buffer_size: int, save_progress_freq: int):
        # parameters
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # ML agent
        self.env = env
        self.env.reset()
        self.behavior_name = list(env.behavior_specs.keys())[0] # assume there's only one behavior
        self.state_size = env.behavior_specs[self.behavior_name].observation_specs[0].shape[0] # assume there's only one observation and the observation is one-dimentional
        self.action_size = env.behavior_specs[self.behavior_name].action_spec.continuous_size # assume only continuous action is used
        self.is_agent_waiting = True

        # training
        self.dqn = dqn
        self.dqn.set_q_networks((self.state_size,), self.action_size)
        self.step = 0
        self.buffer = ReplayBuffer(buffer_size)
        self.prev_time_step: TimeStep = TimeStep(state_size=self.state_size, action_size=self.action_size) 
        self.time_step: TimeStep = TimeStep(state_size=self.state_size, action_size=self.action_size) 
        self.save_progress_freq = save_progress_freq

        # tensorboard
        self.acc_reward = 0
        self.random_action_count = 0
        self.chosen_action_count = 0
        self.episode = 0
        self.action_histogram = np.zeros(self.action_size)
        self.stats_channel = stats_channel

        # debug
        self._logger = logging.getLogger('DQNTrainer')
        self._logger.setLevel(logging.DEBUG)
        file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log', 'DQNTrainer.log')
        self._logger.addHandler(logging.FileHandler(filename=file))

    def start_learning(self, progress_path: str = None):
        if progress_path != None:
            self._resume(progress_path)

        try:
            while not self._done_training():
                self._advance()
        finally:
            self._save_progress()
            self._save_model()
            ftm.print()

    def start_inference(self, model_path: str):
        """
        Do inference using the model.
        Model file/folder must be of SavedModel format.
        """
        # load model and set the model to Q network
        model = tf.keras.models.load_model(model_path, custom_objects={'loss': self.dqn._get_custom_loss()})
        model.summary()
        self.dqn.q_network.model = model

        while True:
            self._advance_inference()

    def _done_training(self) -> bool:
        return self.step > self.max_steps

    def _advance(self):
        # choose action
        if self.dqn.is_random():
            action = self.dqn.get_random_action()
            self.random_action_count += 1
        else:
            action = self.dqn.get_action(self.prev_time_step.state)
            self.chosen_action_count += 1

        # send action to agent in Unity
        action_array = np.empty(
            (
                1, # assume there's only one agent waiting
                self.env.behavior_specs[self.behavior_name].action_spec.continuous_size
            ),
            dtype=np.float32
        )
        action_array[0] = action
        if self.is_agent_waiting:
            self.env.set_actions(behavior_name=self.behavior_name, action=ActionTuple(continuous=action_array))

        # step environment
        self.env.step()
        self.step += 1
        print(self.step)
        self.dqn.count_step()

        # observe game state and store it
        (decision_steps, terminal_steps) = self.env.get_steps(behavior_name=self.behavior_name)

        if len(decision_steps) == 1:
            agent_id = decision_steps.agent_id[0]
            self.prev_time_step.next_state = decision_steps[agent_id].obs[0]
            self.prev_time_step.next_action = action
            self.time_step.state = decision_steps[agent_id].obs[0]
            self.time_step.action = action
            self.time_step.reward = decision_steps[agent_id].reward
            self.time_step.done = 0
            
            self.is_agent_waiting = True

            self.acc_reward += self.time_step.reward
            self.action_histogram[action.argmax()] += 1
        
        if len(terminal_steps) == 1:
            agent_id = terminal_steps.agent_id[0]
            self.prev_time_step.next_state = terminal_steps[agent_id].obs[0]
            self.prev_time_step.next_action = action
            self.time_step.state = terminal_steps[agent_id].obs[0]
            self.time_step.action = action
            self.time_step.reward = terminal_steps[agent_id].reward
            self.time_step.done = 1

            self.is_agent_waiting = False

            self.acc_reward += self.time_step.reward
            self.action_histogram[action.argmax()] += 1
            self.action_histogram /= np.linalg.norm(self.action_histogram)
            self.episode += 1

            # self._logger.debug(self.time_step)

            # TensorBoard
            with self.dqn.train_writer.as_default():
                tf.summary.scalar('training/accumulated reward', self.acc_reward, step=self.step)
                tf.summary.scalar('training/randomness', self.random_action_count / (self.random_action_count + self.chosen_action_count), step=self.step)
                tf.summary.scalar('training/episode', self.episode, step=self.step)
                tf.summary.scalar('game info/agent wins', self.stats_channel.agent_win_count, step=self.step)
                tf.summary.scalar('game info/opponent wins', self.stats_channel.opponent_win_count, step=self.step)
                if self.dqn.action_histogram_in_demonstrations is not None:
                    tf.summary.scalar('game info/action histogram error', np.abs(self.action_histogram - self.dqn.action_histogram_in_demonstrations).mean(), step=self.step)
                    self._logger.debug(f'episode: {self.episode}, histo in demonstrations: {self.dqn.action_histogram_in_demonstrations}, histo in this episode: {self.action_histogram}')
                tf.summary.flush()
            self.acc_reward = 0
            self.random_action_count = 0
            self.chosen_action_count = 0
            self.action_histogram = np.zeros(self.action_size)
        
        self.buffer.append(self.prev_time_step)
        self.prev_time_step = copy(self.time_step)

        # update Q network
        if self.dqn.is_update():
            batch = self.buffer.sample(self.batch_size)
            self.dqn.update(batch)

        # update target Q network
        if self.dqn.is_update_target():
            self.dqn.update_target()

        # save progress
        if self.step % self.save_progress_freq == 0 and self.step != 0:
            self._save_progress()
            self._save_model(dir=os.path.join(DQNProgress.dir, f"{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_step={self.step}"))

    def _advance_inference(self):
        
        # choose action 
        if self.dqn.is_random():
            action = self.dqn.get_random_action()
        else:
            action = self.dqn.get_action(self.prev_time_step.state)

        # send action
        action_array = np.empty(
            (
                1, # assume there's only one agent waiting
                self.env.behavior_specs[self.behavior_name].action_spec.continuous_size
            ),
            dtype=np.float32
        )
        action_array[0] = action
        if self.is_agent_waiting:
            self.env.set_actions(behavior_name=self.behavior_name, action=ActionTuple(continuous=action_array))

        # step env
        self.env.step()
        self.step += 1
        print(self.step)
        self.dqn.count_step()

        # observe game state
        (decision_steps, terminal_steps) = self.env.get_steps(behavior_name=self.behavior_name)

        if len(decision_steps) == 1:
            agent_id = decision_steps.agent_id[0]
            self.prev_time_step.next_state = decision_steps[agent_id].obs[0]
            self.prev_time_step.next_action = action
            self.time_step.state = decision_steps[agent_id].obs[0]
            self.time_step.action = action
            self.time_step.reward = decision_steps[agent_id].reward
            self.time_step.done = 0
            
            self.is_agent_waiting = True
        
        if len(terminal_steps) == 1:
            agent_id = terminal_steps.agent_id[0]
            self.prev_time_step.next_state = terminal_steps[agent_id].obs[0]
            self.prev_time_step.next_action = action
            self.time_step.state = terminal_steps[agent_id].obs[0]
            self.time_step.action = action
            self.time_step.reward = terminal_steps[agent_id].reward
            self.time_step.done = 1

            self.is_agent_waiting = False
        
        self.buffer.append(self.prev_time_step)
        self.prev_time_step = copy(self.time_step)

    def _resume(self, progress_file_path):
        with open(progress_file_path, 'rb') as f:
            progress: DQNProgress = pickle.load(f)
            self.dqn.q_network.model.set_weights(progress.model_weights)
            self.dqn.target_q_network.model.set_weights(progress.target_model_weights)
            self.buffer = progress.replay_buffer
            self.step = progress.step
            self.dqn._step = progress.step
            self.episode = progress.episode
            self.stats_channel.agent_win_count = progress.agent_wins
            self.stats_channel.opponent_win_count = progress.opponent_wins

    def _save_progress(self): 
        progress = DQNProgress(
            model_weights=self.dqn.q_network.model.get_weights(),
            target_model_weights=self.dqn.target_q_network.model.get_weights(),
            replay_buffer=self.buffer,
            step=self.step,
            episode=self.episode,
            agent_wins=self.stats_channel.agent_win_count,
            opponent_wins=self.stats_channel.opponent_win_count
        )
        progress.save()

    def _save_model(self, dir=None):
        now = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        if dir == None:
            dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", now)
        os.makedirs(dir, exist_ok=True)
        self.dqn.q_network.model.save(dir)


def main():
    stats_channel = StatisticalSideChannel()
    env = UnityEnvironment(side_channels=[stats_channel])
    MAX_STEPS = 1000000000
    BATCH_SIZE = 32
    BUFFER_SIZE = 100
    dqn = DQN(
        learning_rate=0.00025,
        hidden_layer_unit=34,
        initial_epsilon=0,
        final_epsilon=0,
        epsilon_decay=0,
        gamma=0.999,
        start_steps=0,
        update_interval=4,
        target_update_interval=10000,
        reward_shaping=True,
        demonstrations_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demonstrations'),
        min_similarity=0.5
    )
    trainer = DQNTrainer(env, dqn, stats_channel, MAX_STEPS, BATCH_SIZE, BUFFER_SIZE, save_progress_freq=BUFFER_SIZE)

    try:
        # trainer.start_learning()
        trainer.start_inference(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', '2023-01-20_02-23-12-20230208T084533Z-001', '2023-01-20_02-23-12'))
    finally:
        env.close()
    
    
    
if __name__ == "__main__":
    main()