import json
import numpy as np

from env import (
    SimpleMarketSimulator,
    DeepMarketMaker
)


def main():

    n_sim = 10
    dim = 2000

    q_grid = np.zeros((dim, n_sim))

    sigma = 0.01
    bid_k = 15.5
    bid_a = 0.80
    ask_k = 15.5
    ask_a = 0.80
    dt = 1.0

    dungeon = SimpleMarketSimulator(dim, sigma, bid_k, bid_a, ask_k, ask_a, dt, s0=100)

    agent = DeepMarketMaker()

    total_reward = 0
    last_total = 0
    for sim in range(0, n_sim):

        dungeon.reset()

        for step in range(dim):

            old_state = dungeon.state  # Store current state

            action_b, action_a = agent.get_next_action(old_state)  # Query agent for the next action
            new_state, reward = dungeon.take_action(action_b, action_a)  # Take action, get new state and reward

            if dungeon.done:
                break

            agent.update(old_state, new_state, action_b, action_a, reward)

            total_reward += reward  # Keep score
            if step % 250 == 0:  # Print out metadata every 250th iteration
                performance = (total_reward - last_total) / 250.0
                print(json.dumps({'step': step, 'performance': performance, 'total_reward': total_reward}))
                last_total = total_reward

        q_grid[:, sim] = dungeon.q

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(q_grid)
    plt.show()

    for q in range(-5, 5):
        action_b, action_a = agent.get_next_action(q)
        print(q, action_b, action_a)


if __name__ == "__main__":
    main()