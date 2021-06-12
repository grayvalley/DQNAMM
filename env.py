import numpy as np
import random
from itertools import product


def fill_prob(lamda, kappa, distance, dt):
    """
    Computes exponential fill probability assuming Poisson arrival
    at the top of the book.
    """
    return lamda * np.exp(-kappa * distance) * dt


class DeterministicMarketMakingSimulator:
    """
    This is a deterministic market making environment. Deterministic in the
    sense that fill probability is always 100%.
    """
    def __init__(self, dim, sigma, bid_k, bid_a, ask_k, ask_a,
                 max_bid_depth, max_ask_depth, dt, inv_pen, rebate, s0=100):

        # price process parameters
        self.sigma = sigma
        self.bid_k = bid_k
        self.bid_a = bid_a
        self.ask_k = ask_k
        self.ask_a = ask_a
        self.max_bid_depth = max_bid_depth
        self.max_ask_depth = max_ask_depth
        self.inventory_penalty = inv_pen
        self.rebate = rebate
        self.s0 = s0
        self.dt = dt
        self.dim = dim

        self.reset()
        
    @property
    def s(self):
        return self.price_process[self.step_idx]

    def reset(self):

        self.action_space = list(
                product(range(self.max_bid_depth+1),
                        range(self.max_ask_depth+1)))
        
        self.q = np.zeros(self.dim)  # vector of inventories
        self.x = np.zeros(self.dim)  # vector of cash
        self.bid_hit = np.zeros(self.dim, dtype=bool) # bid fills
        self.ask_lift = np.zeros(self.dim, dtype=bool) # ask fills

        # simulate mid-price process
        self.price_process = self.s0 + np.cumsum(
                self.sigma * np.sqrt(self.dt)
                * np.random.choice([1, -1],  self.dim))

        self.steps = len(self.price_process)
        self.step_idx = 1
        self.step_idx_max = self.steps-1
        self.done = False

    @property
    def state(self):
        return self.q[self.step_idx-1]


    def inventory_change_reward(self, q_old, q_new):
        """
        Compute reward based on steering of inventory into the right
        direction. If any case, we reward for action that
        steers inventory towards zero.
        """
        
        reward = 0
        
        # Change in the nventory during this period
        dq = q_new - q_old
        
        # If the agent has positive inventory, we want to reward for action
        # that decreases the inventory towards zero.
        if q_old > 0: 
            reward = -dq
                
        # If the agent has negative inventory, we want to reward for action
        # that increases the inventory towards zero.
        elif q_old < 0: 
            reward = dq
                    
        return 10*reward
    
    
    def spread_capture_reward(self, bid_hit, ask_hit):
        """
        Compute reward based on captured spread
        """
        
        reward = 0
        
        # If bid was filled, give a reward for posting liquidity
        reward += bid_hit 
        
        # If ask was filled, give a reward for posting liquidity
        reward += ask_hit
        
        return 0.5 * reward
    
    def step(self, action):
        
        # Get action tuple
        action_tuple = self.action_space[action]
        
        # Bid and ask skews 
        n_bid_skew, n_ask_skew = action_tuple
        
        # Current inventory
        q_old = self.q[self.step_idx-1]
        
        
        # Fill probabilities for quoted spreads
        p_bid_hit = 0
        if n_bid_skew > 0:
            p_bid_hit = 1 #fill_prob(self.bid_a, self.bid_k, spread_bid, self.dt)
            
        p_ask_hit = 0
        if n_ask_skew > 0:
            p_ask_hit =  1# fill_prob(self.ask_a, self.ask_k, spread_ask, self.dt)
                
        # Initialize inventory and cash position for current step
        self.q[self.step_idx] = self.q[self.step_idx - 1]
        self.x[self.step_idx] = self.x[self.step_idx - 1]
            
        # Simulation of bid fill
        bid_hit = np.random.random_sample() <= p_bid_hit
        if bid_hit:
            self.q[self.step_idx] += 1
            
        # Simulation of ask fill
        ask_hit = np.random.random_sample() <= p_ask_hit
        if ask_hit:
            self.q[self.step_idx] -= 1
            
        q_new = self.q[self.step_idx]
        

        # Compute spread change reward
        reward_spread = self.spread_capture_reward(bid_hit, ask_hit)
        
        # Compute inventory change reward
        reward_inventory = self.inventory_change_reward(q_old, q_new)
        
        # Net reward
        net_reward = reward_spread + reward_inventory
        
        print(q_old, q_new, reward_spread, reward_inventory, net_reward)

        if self.step_idx < self.step_idx_max - 1:
            new_state = self.q[self.step_idx]
            self.step_idx += 1
            return new_state, net_reward, False
        else:
            self.done = True
            return None, None, True


