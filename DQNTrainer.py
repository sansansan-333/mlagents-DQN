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

from DQN import DQN, QNetwork, ReplayBuffer, TimeStep, ftm
from DQNProgress import DQNProgress

# This class is dependent on the UFE environment
class DQNTrainer: 
    def __init__(self, env: UnityEnvironment, dqn: DQN, max_steps: int, batch_size: int, buffer_size: int, save_progress_freq: int):
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
        self.is_agent_waiting = False

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

    def _done_training(self) -> bool:
        return self.step > self.max_steps

    def _advance(self):
        # choose action
        if self.dqn.is_random():
            action = self.dqn.get_random_action()
        else:
            action = self.dqn.select_action(self.prev_time_step.state)

        # send action to agent in unity
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
        
        if len(terminal_steps) == 1:
            agent_id = terminal_steps.agent_id[0]
            self.prev_time_step.next_state = terminal_steps[agent_id].obs[0]
            self.prev_time_step.next_action = action
            self.time_step.state = terminal_steps[agent_id].obs[0]
            self.time_step.action = action
            self.time_step.reward = terminal_steps[agent_id].reward
            self.time_step.done = 1
            self._logger.debug(self.time_step)

            self.is_agent_waiting = False
            with self.dqn.train_writer.as_default():
                tf.summary.scalar('accumulated reward', self.acc_reward, step=self.step)
                tf.summary.flush()
            self.acc_reward = 0
        
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


    def _resume(self, progress_file_path):
        with open(progress_file_path, 'rb') as f:
            progress: DQNProgress = pickle.load(f)
            self.dqn.q_network.model.set_weights(progress.model_weights)
            self.dqn.target_q_network.model.set_weights(progress.target_model_weights)
            self.step = progress.step
            self.buffer = progress.replay_buffer
            self.dqn._step = progress.step

    def _save_progress(self): 
        progress = DQNProgress(
            model_weights=self.dqn.q_network.model.get_weights(),
            target_model_weights=self.dqn.target_q_network.model.get_weights(),
            step=self.step,
            replay_buffer=self.buffer
        )
        progress.save()

    def _save_model(self, dir=None):
        now = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        if dir == None:
            dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", now)
        os.makedirs(dir, exist_ok=True)
        self.dqn.q_network.model.save(dir)


def main():
    env = UnityEnvironment()
    dqn = DQN(
        epsilon=0.05,
        gamma=0.999,
        start_steps=100000,
        update_interval=4,
        target_update_interval=10000,
        demonstrations_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demonstrations')
    )
    MAX_STEPS = 1000000000
    BATCH_SIZE = 16
    BUFFER_SIZE = 100000
    trainer = DQNTrainer(env, dqn, MAX_STEPS, BATCH_SIZE, BUFFER_SIZE, save_progress_freq=100000)

    # TODO: wait until game begins
    # while ...

    try:
        trainer.start_learning()
    finally:
        env.close()
    
    
    
if __name__ == "__main__":
    main()