# Import libraries 
import numpy as np 
 
class EpsilonGreedy: 
def  init (self, n_arms, epsilon): 
self.n_arms = n_arms 
self.epsilon = epsilon 
self.counts = np.zeros(n_arms) 
self.values = np.zeros(n_arms) 
 
def select_arm(self): 
if np.random.rand() < self.epsilon: 
return np.random.randint(0, 
self.n_arms) 
else: 
return np.argmax(self.values) 
 
def update(self, chosen_arm, reward): 
self.counts[chosen_arm] += 1

n = self.counts[chosen_arm] 
value = self.values[chosen_arm] 
self.values[chosen_arm] = ((n - 1) / 
n) * value + (1 / n) * reward 
 
# Parameters 
n_arms = 10 
epsilon = 0.1 
n_trials = 1000 
 
# True rewards 
true_means = np.random.randn(n_arms) 
print(f"Actual mean rewards: 
{np.round(true_means, 2)}") 
print(f"Best arm: {np.argmax(true_means)} with 
mean {np.max(true_means):.2f}\n") 
 
# Initialize agent 
agent = EpsilonGreedy(n_arms, epsilon) 
total_reward = 0 
 
# Run trials 
for t in range(n_trials): 
arm_to_pull = agent.select_arm() 
reward = np.random.randn() + 
true_means[arm_to_pull] 
agent.update(arm_to_pull, reward) 
total_reward += reward 
 
print(f"Total Reward: {total_reward:.2f}") 
