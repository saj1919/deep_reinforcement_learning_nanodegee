import random
import numpy as np
import torch
import torch.nn.functional as F
import copy
from collections import namedtuple, deque
from network import Actor, Critic

SEED = 0
TAU = 1e-3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256


class DDPGAgent:
    def __init__(self, action_size, state_size, shared_replay_buffer, memory):
        optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)
        noise_fn = lambda: OUNoise(action_size, SEED)
        memory_fn = lambda: ReplayBuffer(action_size, int(1e6), BATCH_SIZE, SEED, DEVICE)
        actor_network_fn = lambda: Actor(action_size, state_size, (256,128), SEED).to(DEVICE)
        critic_network_fn = lambda: Critic(action_size, state_size, (256,128), SEED).to(DEVICE)
        
        self.seed = SEED
        
        self.actor_local = actor_network_fn()
        self.actor_target = actor_network_fn()
        self.actor_optimizer = optimizer_fn(self.actor_local.parameters())
        
        self.critic_local = critic_network_fn()
        self.critic_target = critic_network_fn()
        self.critic_optimizer = optimizer_fn(self.critic_local.parameters())
        
        self.soft_update(self.critic_local, self.critic_target, 1)
        self.soft_update(self.actor_local, self.actor_target, 1)

        self.noise = noise_fn()
        if shared_replay_buffer:
            self.memory = memory
        else:
            self.memory = memory_fn()
            
        
    def reset(self):
        self.noise.reset()
        
    def act(self, states):            
        states = torch.from_numpy(states).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        actions += self.noise.sample()
        return np.clip(actions, -1, 1)
                
    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences
        
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
    
    
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)
        
    def __len__(self):
        return len(self.memory)
    
    