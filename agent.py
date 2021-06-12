import numpy as np
from tensorflow.keras.models import load_model

from buffer import (
        ReplayBuffer,
        build
)

class DQNAgent:
    
    def __init__(self, alpha, gamma, n_actions, epsilon,
                 batch_size, input_dims, fc1_dims, fc2_dims, epsilon_dec=0.996, 
                 epsilon_end=0.01, mem_size=1000000,
                 fname='dqn_model.h5'):
        
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        
        self.q_eval = build(alpha, n_actions, input_dims, fc1_dims, fc2_dims)
        
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def choose_action(self, state):
        """
        Choose action given state of the game.
        """
        state = state[np.newaxis, :]
        
        # epsilon greedy
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
            
        return action
    
    def learn(self):
        
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)
        
        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)
        q_target = q_eval.copy()
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1) * done
        
        _ = self.q_eval.fit(state, q_target, verbose=0)
        
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
        
    
    def save_model(self):
        self.q_eval.save(self.model_file)
        
    def load_model(self):
        self.q_eval = load_model(self.model_file)