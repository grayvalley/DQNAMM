#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 21:27:55 2021

@author: juha
"""


from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

import numpy as np


class ReplayBuffer:
    
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        
        self.mem_size = max_size
        self.mem_cntr = 0
        self.input_shape = input_shape
        self.discrete = discrete
        dtype = np.int8 if self.discrete else np.float32
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        
    
    def store_transition(self, state, action, reward, state_, done):
        
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, terminal
    

def build(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    
    model = Sequential([
            Dense(fc1_dims, input_shape=(input_dims,)),
            Activation('relu'),
            Dense(fc2_dims),
            Activation('relu'),
            Dense(n_actions)
            ])
    
    model.compile(optimizer=Adam(lr=lr), loss='mse')
    
    return model
