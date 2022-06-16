# TODO check import
from environment.env import RacingEnv
from models.rl.agents import *

class ReinforcementLearningRunner():
    def __init__(self, agent_kwargs, env_kwargs, sim_kwargs):
        # TODO: initialize environment
        self.env = RacingEnv(env_kwargs, sim_kwargs)
        # TODO: initialize agent
        
        # TODO: initialize visual encoder
        pass
    
    def train(self):
        pass
    
    def evaluate(self):
        pass