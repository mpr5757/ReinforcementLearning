import gymnasium as gym
import numpy as np
from collections import defaultdict

#This is for Q 1.1 (non graded testing code)
'''
env = gym.make("Blackjack-v1", render_mode="human") # Initializing environments
observation, info = env.reset()
for _ in range(50):
action = env.action_space.sample() # agent policy that uses the observation and info
observation, reward, terminated, truncated, info = env.step(action)
if terminated or truncated:
observation, info = env.reset()
env.close()
'''

#This is Q 1.2:
env = gym.make("Blackjack-v1", natural = False, sab = False) # Initializing environments

class q_learning_agent():
    def __init__(self, alpha, gamma, epsilon, decreaseEpsilon):
        # we have 2 things in our action space, hit or stand. We want the q values of both to be initialized to 0 for each state, for all possible states. 
        # To do this, we use a "shortcut function" lambda to set the 0s for our action space. We use defaultdict to set a value for a state we haven't seen before, rather than return error.
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n)) 
        self.alpha = alpha #This is the learning rate
        self.gamma = gamma #This is the discount factor
        self.epsilon = epsilon #This is for the epsilon greedy policy
        #From the lecture slide RL2, we ae going to decrease epsilon over time. 
        self.decreaseEpsilon = decreaseEpsilon



    # function for exploration. This uses espilon greedy policy from RL2 lecture slide. 
    def choose_action(self, observation):
        # If we are in the exploration phase, with small probability epsilon, we act randomly
        if np.random.rand() < self.epsilon:
            action = env.action_space.sample()
        # Else with large probability 1 - epsilon, we act on current policy (greedy)
        else:
            action = np.argmax(self.q_table[observation])
        return action
    
    # function for updating the q table after our agent takes an action
    def q_value_update(self, action):
        prev_observation = action[0]
        resulting_observation = action[1]
        reward = action[2]
        terminated = action[3]
        action_taken = action[4]

        # If the deal is over, then set the maxQk to 0.
        if terminated == True:
            maxQk = 0
        else:
            maxQk = np.max(self.q_table[resulting_observation])

        #We need a function to actually decrease epsilon over time.
        #Also going to place a lower bound for epsilon so it doesn't keep decreasing below 0. 
        def decreaseEpsilon(self):
            self.epsilon = max(0.001, self.epsilon - self.decreaseEpsilon)

        # The Q learning equation is: Q_(s0,a0) <-- (1 - alpha)*Q_(s0,a0) + alpha*(reward + gamma * max_a' Q_(s1,a1))
        # The part (reward + gamma * max_a' Q_(s1,a1)) is known as the sample (from lecture slide RL1). We set up the variables here:
        sample = reward + self.gamma * maxQk
        #We finish the Q learning equation here:
        self.q_table[prev_observation][action_taken] = (1 - self.alpha) * self.q_table[prev_observation][action_taken] + self.alpha * sample



# create the agent. The following values are tweaked based on running the results multiple times.
alpha = 0.2
gamma = 0.9
epsilon = 0.1
episodes = 10000
decreaseEpsilon = epsilon/(episodes/3)
agent = q_learning_agent(alpha, gamma, epsilon, decreaseEpsilon)

for n in range(episodes):
    
    game_finished = False
    
    '''
        GAME START
        WHILE LOOP IS UNTIL THE GAME IS OVER
    '''
    observation, info = env.reset() # reset the environment (aka start)
    while not game_finished:
        # choose an action using our epsilon greedy policy
        action = agent.choose_action(observation)

        # take the action and store the results in four variables.
        resulting_observation, reward, terminated, truncated, info = env.step(action)

        # store the action taken in a tuple and send it into our q vaule update funciton to update the q table.
        action_taken = (observation, resulting_observation, reward, terminated, action)
        agent.q_value_update(action_taken)

        # make the new observation state the current observation state 
        # and repeat the while loop until the game is over
        observation = resulting_observation

        if terminated or truncated:
            game_finished = True 


