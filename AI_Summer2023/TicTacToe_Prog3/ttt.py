import numpy as np
import random

# Initialize the Q-values
Q = np.zeros((3**9, 9))

# Define the epsilon-greedy action selection parameters
epsilon = 0.1
delta_epsilon = 0.01
min_epsilon = 0.1

# Define the discount factor
gamma = 0.9

# Convert board state to an index
def state_to_index(state):
    index = 0
    for i in range(9):
        index += state[i] * (3**i)
    return index

# Perform epsilon-greedy action selection
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(legal_moves(state))
    else:
        return np.argmax(Q[state_to_index(state)])

# Get legal moves for a given state
def legal_moves(state):
    return [i for i in range(9) if state[i] == 0]

# Perform a move and return the new state
def perform_action(state, action, player):
    new_state = state.copy()
    new_state[action] = player
    return new_state

# Check if the game is over and return the reward
def check_goal_state(state, player):
    winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
                            (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    for comb in winning_combinations:
        if state[comb[0]] == state[comb[1]] == state[comb[2]] == player:
            return 1
    if 0 not in state:
        return 0.5  # Draw
    return 0

# Training
def train(num_epochs):
    global epsilon

    for epoch in range(num_epochs):
        state = np.zeros(9, dtype=int)
        done = False

        while not done:
            index = state_to_index(state)
            action = choose_action(state)
            new_state = perform_action(state, action, 1)
            reward = check_goal_state(new_state, 1)

            if not reward:
                opponent_action = random.choice(legal_moves(new_state))
                new_state = perform_action(new_state, opponent_action, -1)
                reward = check_goal_state(new_state, -1)

            Q[index, action] += 0.1 * (reward + gamma * np.max(Q[state_to_index(new_state)]) - Q[index, action])

            state = new_state
            done = reward != 0

        epsilon = max(min_epsilon, epsilon - delta_epsilon)

# Testing against a random opponent
def test(num_games):
    total_score = 0

    for _ in range(num_games):
        state = np.zeros(9, dtype=int)
        done = False

        while not done:
            index = state_to_index(state)
            action = np.argmax(Q[index])
            state = perform_action(state, action, 1)
            reward = check_goal_state(state, 1)

            if not reward:
                opponent_action = random.choice(legal_moves(state))
                state = perform_action(state, opponent_action, -1)
                reward = check_goal_state(state, -1)

            done = reward != 0

        if reward == 1:
            total_score += 1
        elif reward == 0.5:
            total_score += 0.5

    return total_score

# Main loop
def main():
    train(num_epochs=10000)
    score = test(num_games=10)
    print("Total score against baseline opponent:", score)

if __name__ == "__main__":
    main()
