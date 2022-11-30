from DQN import ReplayBuffer
import pickle
import datetime
import os

class DQNProgress:
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress')

    def __init__(self, model_weights, target_model_weights, step: int, replay_buffer: ReplayBuffer):
        self.model_weights = model_weights
        self.target_model_weights = target_model_weights
        self.step = step
        self.replay_buffer = replay_buffer

    def save(self):
        now = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = now + f'_step={self.step}.pickle'
        with open(os.path.join(DQNProgress.dir, file_name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def discard_progress_files():
        '''
        Discard files containing progress info.
        '''
        os.makedirs(DQNProgress.dir, exist_ok=True)
        for file in os.listdir(DQNProgress.dir):
            path = os.path.join(DQNProgress.dir, file)
            if file.endswith('.pickle'):
                os.remove(path)
