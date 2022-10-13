from random import random
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from copy import copy
import numpy as np
import os
import random
import logging
import datetime

from DQN import DQN, QNetwork, ReplayBuffer, TimeStep, ftm

# This class is dependent on the UFE environment
class DQNTrainer: 
    def __init__(self, env: UnityEnvironment, dqn: DQN, max_steps: int, batch_size: int, buffer_size: int):
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

        # debug
        self._logger = logging.getLogger('DQNTrainer')
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(logging.FileHandler(filename='DQNTrainer.log'))

    def start_learning(self):
        try:
            while not self._done_training():
                self._advance()
        finally:
            ftm.print()
            self._save_model()

    def _done_training(self) -> bool:
        return self.step > self.max_steps

    def _advance(self):
        # choose action
        if self.dqn.is_random():
            action = self._get_random_action()
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
        ftm.start('env.step')
        self.env.step()
        self.step += 1
        self.dqn.count_step()
        ftm.end('env.step')

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
        # self._logger.debug(self.prev_time_step)
        self.prev_time_step = copy(self.time_step)

        # update Q network
        if self.dqn.is_update():
            batch = self.buffer.sample(self.batch_size)
            self.dqn.update(batch)

        # update target Q network
        if self.dqn.is_update_target():
            self.dqn.update_target()

    def _get_random_action(self) -> np.ndarray:
        action = np.empty(self.action_size)
        for i in range(self.action_size):
            action[i] = random.random()
        return action

    def _save_model(self):
        now = datetime.datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
        dir = os.path.dirname(__file__) + f"/model/{now}"
        os.makedirs(dir)
        self.dqn.q_network.model.save(dir)


def main():
    env = UnityEnvironment()
    dqn = DQN(
        epsilon=0.01,
        gamma=0.99,
        start_steps=100000,
        update_interval=4,
        target_update_interval=1000
    )
    MAX_STEPS = 1000000000
    BATCH_SIZE = 32
    BUFFER_SIZE = 100000
    trainer = DQNTrainer(env, dqn, MAX_STEPS, BATCH_SIZE, BUFFER_SIZE)

    # TODO: wait until game begins
    # while ...

    try:
        trainer.start_learning()
    finally:
        env.close()
    
    
    
if __name__ == "__main__":
    main()