import numpy as np
import random
import tensorflow as tf


class SimpleMarketSimulator:

    def __init__(self, dim, sigma, bid_k, bid_a, ask_k, ask_a, dt, s0=100):

        # price process parameters
        self.sigma = sigma
        self.bid_k = bid_k
        self.bid_a = bid_a
        self.ask_k = ask_k
        self.ask_a = ask_a
        self.s0 = s0
        self.dt = dt
        self.dim = dim

        self.reset()

    @property
    def s(self):
        return self.price_process[self.step_idx]

    def reset(self):

        self.q = np.zeros(self.dim)      # vector of inventories
        self.x = np.zeros(self.dim)      # vector of cash
        self.bid_hit = np.zeros(self.dim, dtype=bool)
        self.ask_lift = np.zeros(self.dim, dtype=bool)

        # simulate random price change
        self.price_process = self.s0 + np.cumsum(self.sigma * np.sqrt(self.dt) * np.random.choice([1, -1],  self.dim))

        self.steps = len(self.price_process)    # total number of simulation steps
        self.step_idx = 1
        self.step_idx_max = self.steps-1
        self.done = False

    @property
    def state(self):
        return self.q[self.step_idx-1]

    def take_action(self, n_bid_skew, n_ask_skew):

        tick_size = 0.01
        spread_bid = n_bid_skew * tick_size
        spread_ask = n_ask_skew * tick_size

        price_bid = self.s - spread_bid
        price_ask = self.s + spread_ask

        # fill probability for quoted spreads
        p_bid_hit = self.bid_a * np.exp(-self.bid_k * spread_bid) * self.dt
        p_ask_hit = self.ask_a * np.exp(-self.ask_k * spread_ask) * self.dt

        self.q[self.step_idx] = self.q[self.step_idx - 1]
        self.x[self.step_idx] = self.x[self.step_idx - 1]

        q_old = self.q[self.step_idx]

        # simulation of bid fill
        bid_hit = random.random() < p_bid_hit
        if bid_hit:
            self.q[self.step_idx] += 1
            self.x[self.step_idx] -= price_bid

        # simulation of ask fill
        ask_hit = random.random() < p_ask_hit
        if ask_hit:
            self.q[self.step_idx] -= 1
            self.x[self.step_idx] += price_ask

        q_new = self.q[self.step_idx]

        mid_old = self.price_process[self.step_idx]
        mid_next = self.price_process[self.step_idx + 1]

        ask_capture = price_ask - mid_next
        bid_capture = mid_next - price_bid
        spread_pnl = ask_capture * ask_hit + bid_capture * bid_hit

        mtm = mid_next*q_new - mid_old*q_old

        reward = spread_pnl + mtm - 0.1 * np.abs(self.q[self.step_idx])**2

        if self.step_idx < self.step_idx_max - 1:
            new_state = self.q[self.step_idx]
            # move index to next simulated price
            self.step_idx += 1
            return new_state, reward
        else:
            # end of simulation
            self.done = True
            return None, None


class DeepMarketMaker:

    def __init__(self, learning_rate=0.05, discount=0.95, exploration_rate=0.25, iterations=10000):
        self.learning_rate = learning_rate
        self.discount = discount  # How much we appreciate future reward over current
        self.exploration_rate = exploration_rate  # Initial exploration rate
        self.exploration_delta = 1.0 / iterations  # Shift from exploration to exploitation

        self.input_count = 1

        self.n_bids = 8
        self.n_asks = 8
        self.n_depths = self.n_bids + self.n_asks

        self.output_count = self.n_bids + self.n_asks

        self.session = tf.compat.v1.Session()
        self.define_model()
        self.session.run(self.initializer)

    def define_model(self):

        tf.compat.v1.disable_eager_execution()

        # Input is 2-dimensional, due to possibility of batched training data
        # NOTE: In this example we assume no batching.
        self.model_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, self.input_count])

        # 8 hidden neurons per layer
        layer_size = 8

        # Two hidden layers of 8 neurons with sigmoid activation initialized to zero for stability
        nn = tf.compat.v1.layers.dense(
            self.model_input, units=layer_size, activation=tf.sigmoid,
            kernel_initializer=tf.constant_initializer(np.zeros((self.input_count, layer_size))))

        nn = tf.compat.v1.layers.dense(nn, units=layer_size, activation=tf.sigmoid)

        # Output is 2-dimensional, due to possibility of batched training data
        # NOTE: In this example we assume no batching.
        self.model_output = tf.compat.v1.layers.dense(nn, units=self.output_count)

        # This is for feeding training output (a.k.a ideal target values)
        self.target_output = tf.compat.v1.placeholder(shape=[None, self.output_count], dtype=tf.float32)
        # Loss is mean squared difference between current output and ideal target values
        loss = tf.losses.mean_squared_error(self.target_output, self.model_output)
        # Optimizer adjusts weights to minimize loss, with the speed of learning_rate
        self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
        # Initializer to set weights to initial values
        self.initializer = tf.compat.v1.global_variables_initializer()

    # Ask model to estimate Q value for specific state (inference)
    def get_Q(self, state):
        # Model input: Single state represented by array of single item (state)
        # Model output: Array of Q values for single state
        q_values = self.session.run(self.model_output, feed_dict={self.model_input: [[state]]})
        return q_values[0]

    def greedy_action(self, state):
        q_values = self.get_Q(state)

        q_values_bid = q_values[0:self.n_bids]
        q_values_ask = q_values[self.n_bids:self.n_depths]
        q_b = np.argmax()
        q_a = np.argmax(q_values[self.n_bids:self.n_depths])

        return q_b, q_a

    def random_action(self):
        q_b = np.random.randint(0, self.n_bids)
        q_a = np.random.randint(self.n_bids, self.n_depths)
        return q_b, q_a

    def get_next_action(self, state):
        if random.random() > self.exploration_rate:  # Explore (gamble) or exploit (greedy)
            return self.greedy_action(state)
        else:
            return self.random_action()

    def train(self, old_state, action_b, action_a, reward, new_state):

        # Ask the model for the Q values of the old state (inference)
        old_state_Q_values = self.get_Q(old_state)

        # Ask the model for the Q values of the new state (inference)
        new_state_Q_values = self.get_Q(new_state)

        new_state_Q_values_bid = new_state_Q_values[0:self.n_bids]
        new_state_Q_values_ask = new_state_Q_values[self.n_bids:self.n_depths]

        # Real Q value for the action we took. This is what we will train towards.
        old_state_Q_values[action_b] = reward + self.discount * np.amax(new_state_Q_values_bid)
        old_state_Q_values[action_a] = reward + self.discount * np.amax(new_state_Q_values_ask)

        # Setup training data
        training_input = [[old_state]]
        target_output = [old_state_Q_values]
        training_data = {self.model_input: training_input, self.target_output: target_output}

        # Train
        self.session.run(self.optimizer, feed_dict=training_data)

    def update(self, old_state, new_state, action_b, action_a, reward):
        # Train our model with new data
        self.train(old_state, action_b, action_a, reward, new_state)

        # Finally shift our exploration_rate toward zero (less gambling)
        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta
