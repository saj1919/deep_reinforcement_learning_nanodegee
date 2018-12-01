import torch
from agent import DDPGAgent, OUNoise, ReplayBuffer
import numpy as np

SEED = 0
TAU = 1e-3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
UPDATE_EVERY = 4
BATCH_SIZE = 256
DISCOUNT = 0.99



class MultiAgent:
    def __init__(self, action_size, state_size, shared_replay_buffer, num_agents):
        self.shared_replay_buffer = shared_replay_buffer
        memory_fn = lambda: ReplayBuffer(action_size, int(1e6), BATCH_SIZE, SEED, DEVICE)
        
        memory = None
        if shared_replay_buffer:
            self.memory = memory_fn()
            memory = self.memory
        
        self.ddpg_agents = [DDPGAgent(action_size, state_size, shared_replay_buffer, memory) for _ in range(num_agents)]
        self.t_step = 0
     
    def reset(self):
        for agent in self.ddpg_agents:
            agent.reset()
    
    def act(self, all_states):
        actions = [agent.act(np.expand_dims(states, axis=0)) for agent, states in zip(self.ddpg_agents, all_states)]
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                for agent in self.ddpg_agents:
                    if self.shared_replay_buffer:
                        experiences = self.memory.sample()
                    else:
                        experiences = agent.memory.sample()
                    agent.learn(experiences, DISCOUNT)



