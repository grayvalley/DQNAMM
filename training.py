import numpy as np

from env import (
    SimpleMarketMakingSimulator
)

from agent import (
    DQNAgent
)


def main():
    
    # Number of episodes
    n_episodes = 20
    
    # Simulation environment
    env = SimpleMarketMakingSimulator(
            1000, 0.1, 0.80, 0.80, 50, 50, 10, 10, 1, 0.01, 2.5/10000, 100)
    
    # The DQN learning agent
    agent = DQNAgent(
            gamma = 0.99, 
            epsilon = 1.0, 
            alpha = 0.0005, 
            input_dims = 1,
            fc1_dims = 5, fc2_dims = 5,
            n_actions = len(env.action_space), 
            mem_size = 1000000, 
            batch_size = 25, 
            epsilon_end = 0.01)
    
    scores = []
    eps_history = []
    for i in range(n_episodes):
        done = False
        score = 0
        
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
            agent.learn()
            
            # Set next observation 
            observation = observation_
            
            # Increment total reward
            score += reward
            
        eps_history.append(agent.epsilon)
        scores.append(score)
        
        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)
        
        if i % 10 == 0 and i > 0:
            agent.save_model()
 

if __name__ == '__main__':
    main()