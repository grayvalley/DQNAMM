import numpy as np

from env import (
    DeterministicMarketMakingSimulator
)

from agent import (
    DQNAgent
)


def main():
    
    # Number of episodes
    n_episodes = 100
    
    # Simulation environment
    env = DeterministicMarketMakingSimulator(
            10000, 0.1, 0.80, 0.80, 50, 50, 2, 2, 1, 0.01, 2.5/10000, 100)
    
    
    # Max exploration probability
    epsilon_max = 0.25
    
    # Min exploration probability
    epsilon_min = 0.1
    
    # Learning rate
    learning_rate = 0.001
    
    
    # The DQN learning agent
    agent = DQNAgent(
            gamma = 0.99,
            epsilon = epsilon_max,
            epsilon_end = epsilon_min,
            alpha = learning_rate,
            input_dims = 1,
            fc1_dims = 8,
            fc2_dims = 8,
            n_actions = len(env.action_space), 
            mem_size = 1000000, 
            batch_size = 2500,
            epsilon_dec=0.999999
            )
    
    scores = []
    eps_history = []
    for i in range(n_episodes):
        
        done = False
        score = 0
        loopcnt = 0
        
        # Reset the market
        env.reset()
        
        # Get first observation of state variables
        observation = env.state
        
        while not done:
            
            # Choose next epsilon greedy action
            action = agent.choose_action(
                    np.array([observation], dtype=np.int64))
            
            # Step step forward in time
            observation_, reward, done = env.step(action)
            
            if done:
                break;
            
            # Put to buffer
            agent.remember(observation, action, reward, observation_, done)
            
            # Run learning
            if loopcnt % 100 == 0:
                agent.learn()
            
            # Set next observation 
            observation = observation_
            
            # Increment total reward
            score += reward
            loopcnt += 1
            
        eps_history.append(agent.epsilon)
        scores.append(score)
        
        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)
        
        agent.save_model()
        

if __name__ == '__main__':
    main()