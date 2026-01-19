import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --------------------
# Maze definition
# --------------------
maze = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
])

start = (0, 0)
goal = (9, 9)

# --------------------
# Hyperparameters
# --------------------
num_episodes = 5000
alpha = 0.1
gamma = 0.9
epsilon = 0.5
epsilon_min = 0.01
epsilon_decay = 0.995

reward_wall = -10
reward_goal = 50
reward_step = -1

actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # left, right, up, down

Q = np.zeros(maze.shape + (len(actions),))

# --------------------
# Helper functions
# --------------------
def is_valid(pos):
    r, c = pos
    return (
        0 <= r < maze.shape[0] and
        0 <= c < maze.shape[1] and
        maze[r, c] == 0
    )

def choose_action(state):
    if np.random.random() < epsilon:
        return np.random.randint(len(actions))
    return np.argmax(Q[state])

# --------------------
# Training loop
# --------------------
rewards_all_episodes = []

for episode in range(num_episodes):
    state = start
    total_reward = 0
    done = False

    while not done:
        action_idx = choose_action(state)
        move = actions[action_idx]
        next_state = (state[0] + move[0], state[1] + move[1])

        # Invalid move → penalty, stay in place
        if not is_valid(next_state):
            reward = reward_wall
            next_state = state

        elif next_state == goal:
            reward = reward_goal
            done = True

        else:
            reward = reward_step

        old_q = Q[state][action_idx]
        next_max = np.max(Q[next_state])

        Q[state][action_idx] = old_q + alpha * (
            reward + gamma * next_max - old_q
        )

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_all_episodes.append(total_reward)

# --------------------
# Extract optimal path
# --------------------
def get_optimal_path(Q, start, goal, actions, maze, max_steps=200):
    path = [start]
    state = start
    visited = set()

    for _ in range(max_steps):
        if state == goal:
            break

        visited.add(state)
        best_action = np.argmax(Q[state])
        move = actions[best_action]
        next_state = (state[0] + move[0], state[1] + move[1])

        if not is_valid(next_state) or next_state in visited:
            break

        state = next_state
        path.append(state)

    return path

optimal_path = get_optimal_path(Q, start, goal, actions, maze)

# --------------------
# Visualization
# --------------------
def plot_maze_with_path(path):
    cmap = ListedColormap(['#eef8ea', '#a8c79c'])

    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap=cmap)

    plt.scatter(start[1], start[0], c='#81c784', s=200,
                edgecolors='black', label='Start', zorder=5)
    plt.scatter(goal[1], goal[0], c='#388e3c', s=300, marker='*',
                edgecolors='black', label='Goal', zorder=5)

    rows, cols = zip(*path)
    plt.plot(cols, rows, c='#60b37a', linewidth=4,
             label='Learned Path', zorder=4)

    plt.title('Q-Learning: Robot Maze Navigation')
    plt.gca().invert_yaxis()
    plt.xticks(range(maze.shape[1]))
    plt.yticks(range(maze.shape[0]))
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("maze_result.png", dpi=300)
    plt.close()

plot_maze_with_path(optimal_path)
