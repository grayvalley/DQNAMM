import numpy as np
import random
import tensorflow as tf

from itertools import product


def fill_prob(lamda, kappa, distance, dt):
    """
    Computes exponential fill probability assuming Poisson arrival
    at the top of the book.
    """
    return lamda * np.exp(-kappa * distance) * dt


class SimpleMarketMakingSimulator:

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

    def step(self, action):
        
        # Get action tuple
        action_tuple = self.action_space[action]
        
        # Bid and ask skews 
        n_bid_skew, n_ask_skew = action_tuple
        
        tick_size = 0.01
        
        # Actual spreads
        spread_bid = n_bid_skew * tick_size
        spread_ask = n_ask_skew * tick_size
        
        # Quoted prices
        price_bid = self.s - spread_bid
        price_ask = self.s + spread_ask

        # Fill probabilities for quoted spreads
        p_bid_hit = 0
        if n_bid_skew > 0:
            p_bid_hit = fill_prob(self.bid_a, self.bid_k, spread_bid, self.dt)
            
        p_ask_hit = 0
        if n_ask_skew > 0:
            p_ask_hit = fill_prob(self.ask_a, self.ask_k, spread_ask, self.dt)

        self.q[self.step_idx] = self.q[self.step_idx - 1]
        self.x[self.step_idx] = self.x[self.step_idx - 1]
            
        # Simulation of bid fill
        bid_hit = random.random() < p_bid_hit
        if bid_hit:
            self.q[self.step_idx] += 1
            self.x[self.step_idx] -= price_bid

        # Simulation of ask fill
        ask_hit = random.random() < p_ask_hit
        if ask_hit:
            self.q[self.step_idx] -= 1
            self.x[self.step_idx] += price_ask

        q_new = self.q[self.step_idx]

        mid_next = self.price_process[self.step_idx + 1]

        ask_capture = price_ask - mid_next
        bid_capture = mid_next - price_bid
        
        spread_pnl = 0
        spread_pnl += (ask_capture + self.rebate) * ask_hit
        spread_pnl += (bid_capture + self.rebate) * bid_hit
        
        inventory_penalty = tick_size * np.abs(q_new)**2
        
        net_reward = spread_pnl - inventory_penalty

        if self.step_idx < self.step_idx_max - 1:
            new_state = self.q[self.step_idx]
            self.step_idx += 1
            return new_state, net_reward, False
        else:
            self.done = True
            return None, None, True


