from environment.env import RacingEnv

class ReinforcementLearningRunner():
    def __init__(self, env_kwargs, sim_kwargs):
        # TODO: initialize environment
        self.env = RacingEnv(env_kwargs, sim_kwargs)
        # TODO: initialize agent
        
        # TODO: initialize visual encoder
        pass
    
    def train(self):
        pass
    
    def evaluate(self):
        pass