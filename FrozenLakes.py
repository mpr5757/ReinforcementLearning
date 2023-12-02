import gymnasium as gym
from collections import defaultdict
import numpy as np

# Initialize the environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

#This is from Q2.1 (Ungraded)
'''
import gymnasium as gym
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", render_mode="human", is_slippery=True, ) #initialization
observation, info = env.reset()
for _ in range(50):
    action = env.action_space.sample() # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
env.close()
'''

'''
    Q2.2: Random Policy
'''
def random_policy():
    # Initialize an empty list to store the training data
    training_data = []
    # Loop over 1000 episodes
    for n in range(1000):
        observation, info = env.reset()
        while True:
            # Sample a random action
            action = env.action_space.sample()
            # Execute the action
            new_observation, reward, terminated, truncated, info = env.step(action)
            # Append the data to the training data
            training_data.append((observation, action, reward, new_observation))
            observation = new_observation
            # If the episode is terminated or truncated, break the loop
            if terminated or truncated:
                break
    return training_data

'''
    Q2.2: Estimating the Transition and Reward Functions
'''
def estimate_model(training_data):
    # Estimate the transition function T(s, a, s') and reward function R(s, a, s')
    T_counts = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    R_sum = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    count_sa = np.zeros((env.observation_space.n, env.action_space.n))

    for data in training_data:
        state, action, reward, next_state,  = data
        T_counts[state, action, next_state] += 1
        R_sum[state, action, next_state] += reward
        count_sa[state, action] += 1

    # Avoid division by zero
    count_sa[count_sa == 0] = 1

    # Normalize to get probabilities
    T = T_counts / count_sa[:, :, np.newaxis]
    R = R_sum / count_sa[:, :, np.newaxis]

    return T, R

'''
Q2.3: Value Iteration
'''
class value_and_policy:
    def __init__(self, gamma, state_size, action_size, T, R, epsilon=1e-6, episodes=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma # Discount factor
        self.T = T  # Transition function
        self.R = R  # Reward function
        self.epsilon = epsilon  # Convergence threshold
        self.episodes = episodes  # Maximum number of iterations
        self.values = np.zeros(self.state_size) #V_0(s) is always initialized to all 0
        self.policy = np.random.randint(0, self.action_size, self.state_size) #we initialize a random policy

        #self.reset_values()

    def value_iteration(self):
        # The formula for value iteration is: V_k+1(s) <-- max_a sum_s' T(s, a, s') * (R(s, a, s') + gamma * V_k(s'))
        new_values = np.zeros(self.state_size) #V_0(s) is always initialized to all 0

        for state in range(self.state_size):
            V_kplus1 = -float('inf') #initial value 

            for action in range(self.action_size):
                #we apply the formula here, which is sum of T(s, a, s') * (R(s, a, s') + gamma * V_k(s'))
                #The : is used to select all possible next states when transitioning from state s to state s' under action a
                value = np.sum(self.T[state, action, :] * (self.R[state, action, :] + self.gamma * self.values))
                #This part of the formula works with the max_a part of the equation
                if value > V_kplus1:
                    V_kplus1 = value

            #once we have the max value, we update the states
            new_values[state] = V_kplus1

        # Check convergence
        result = np.max(np.abs(new_values - self.values))
        self.values = new_values

        return result
    
    '''
    Q2.4: Policy Extraction
    '''

    def policy_extraction(self):
        #Formula taken from UC Berkeley's CSS188 notes on MDPs. 
        # This is a one step lookahead. We use the formula: pi(s) <-- argmax_a sum_s' T(s, a, s') * (R(s, a, s') + gamma * V(s'))
        pi_val = np.zeros(self.state_size) #pi_i+1 is first initialized to 0

        for state in range(self.state_size): 
            #we apply the formula here, which uses V(k) from the value iteration function
            #The : is used to select all possible actions and next states under our policy pi
            #The axis=1 is used to go along each row (state) of a 2d matrix (col is action) in our summation, for giving us the max_a part of the equation
            value = np.sum(self.T[state, :, :] * (self.R[state, :, :] + self.gamma * self.values), axis=1)
            pi_val[state] = np.argmax(value)
        return pi_val


'''
    Q2.2 Testing
'''
training = random_policy()
T_est, R_est = estimate_model(training)


'''
    Q2.3 Testing
'''
valueAndPolicy = value_and_policy(
    gamma=0.9,
    state_size=env.observation_space.n,
    action_size=env.action_space.n,
    T=T_est,
    R=R_est,
    epsilon=0.01,  
    episodes=1000  
)

# Perform value iteration
for iteration in range(valueAndPolicy.episodes):
    V_kplus1 = valueAndPolicy.value_iteration()

    # Check for convergence
    if V_kplus1 < valueAndPolicy.epsilon:
        break

'''
    Q2.4 Testing - Policy Extraction
'''

# Perform policy extraction
extractedPolicy = valueAndPolicy.policy_extraction()


'''
Q2.5: Acting Inside Frozen Lake policy
'''
observation, info = env.reset()
for _ in range(50):
    action = extractedPolicy[observation] #We want this to be chosen according to policy from policy extraction
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
env.close()

