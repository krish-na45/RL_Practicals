import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
n_states = 16
n_actions = 4   # 0=left, 1=right, 2=up, 3=down
goal_state = 15

Q_table = np.zeros((n_states, n_actions))

learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
epochs = 1000

# Get next state based on action
def get_next_state(state, action):
    row, col = divmod(state, 4)

    if action == 0 and col > 0:       # left
        col -= 1
    elif action == 1 and col < 3:     # right
        col += 1
    elif action == 2 and row > 0:     # up
        row -= 1
    elif action == 3 and row < 3:     # down
        row += 1

    return row * 4 + col

# Choose action (epsilon-greedy)
def choose_action(state):
    if random.uniform(0, 1) < exploration_prob:
        return random.randint(0, n_actions - 1)  # explore
    else:
        return np.argmax(Q_table[state])         # exploit

# Reward function
def get_reward(state):
    return 1 if state == goal_state else 0

# Training loop
for episode in range(epochs):
    current_state = 0
    done = False

    while not done:
        action = choose_action(current_state)
        next_state = get_next_state(current_state, action)
        reward = get_reward(next_state)

        # Q-learning update
        old_q = Q_table[current_state, action]
        next_max = np.max(Q_table[next_state])

        new_q = old_q + learning_rate * (reward + discount_factor * next_max - old_q)
        Q_table[current_state, action] = new_q

        current_state = next_state

        if current_state == goal_state:
            done = True

print("Q-learning training complete.")
print("Learned Q-table:")
print(Q_table)

# Visualization
q_values_grid = np.max(Q_table, axis=1).reshape((4, 4))

plt.figure(figsize=(6, 6))
plt.imshow(q_values_grid, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Q-value')
plt.title('Learned Q-values for Each State')

plt.xticks(np.arange(4), ['0', '1', '2', '3'])
plt.yticks(np.arange(4), ['0', '1', '2', '3'])

for i in range(4):
    for j in range(4):
        plt.text(j, i, f'{q_values_grid[i, j]:.2f}',
                 ha='center', va='center', color='black')

plt.gca().invert_yaxis()
plt.grid(True)
plt.show()
