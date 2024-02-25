import numpy as np
import random
import matplotlib.pyplot as plt

# Function to convert the board state to an integer for indexing the Q-matrix
def state_to_int(state):
    state_int = 0
    for i, cell in enumerate(state):
        if cell == 'X':
            state_int += 3 ** i
        elif cell == 'O':
            state_int += 2 * (3 ** i)
    return state_int

# Function to choose an action using epsilon-greedy policy
def choose_action(state, epsilon, Q):
    empty_cells = [i for i, cell in enumerate(state) if cell == ' ']
    if random.uniform(0, 1) < epsilon and empty_cells:
        return random.choice(empty_cells)
    else:
        state_int = state_to_int(state)
        return np.argmax(Q[state_int, :])

# Function to update Q-matrix using Q-learning
def update_q_matrix(state, action, next_state, reward, learning_rate, discount_factor, Q):
    state_int = state_to_int(state)
    next_state_int = state_to_int(next_state)
    max_next_q = np.max(Q[next_state_int, :])
    Q[state_int, action] += learning_rate * (reward + discount_factor * max_next_q - Q[state_int, action])

# Function to check if a player has won
def check_win(state, player):
    winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                            (0, 3, 6), (1, 4, 7), (2, 5, 8),
                            (0, 4, 8), (2, 4, 6)]

    for combo in winning_combinations:
        if all(state[i] == player for i in combo):
            return True
    return False

# Function to play a game against a random opponent
def play_game(epsilon, Q, agent_turn=True):
    board = [' '] * 9
    while True:
        if agent_turn:
            action = choose_action(board, epsilon, Q)
            player = 'X'
        else:
            valid_moves = [i for i, cell in enumerate(board) if cell == ' ']
            action = random.choice(valid_moves)
            player = 'O'
        
        board[action] = player
        
        if check_win(board, player):
            return player
        if ' ' not in board:
            return 'Draw'
        
        agent_turn = not agent_turn

# Function to display the tic-tac-toe board
def display_board(board):
    for i in range(0, 9, 3):
        print(board[i], '|', board[i+1], '|', board[i+2])
        if i < 6:
            print('-' * 9)
    print('')

def main():
    # Initialize Q-matrix
    Q = np.zeros((3**9, 9))  # 3^9 possible states, 9 possible actions

    # Hyperparameters
    epsilon = 0.1
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon_decay = 0.9
    epsilon_min = 0.1

    # ... (rest of the functions remain unchanged)

    # Training and Testing
    num_epochs = 10
    epsilon_decay_interval = 1000
    board = [' '] * 9
    test_epochs = 1000
    scores = []

    for epoch in range(num_epochs):
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Play a game
        while True:
            display_board(board)
            # Player X's turn
            action_X = choose_action(board,epsilon, Q)
            board[action_X] = 'X'

            if check_win(board, 'X'):
                update_q_matrix(board, action_X, board, 1, learning_rate, discount_factor, Q)
                break
            
            if ' ' not in board:
                update_q_matrix(board, action_X, board, 0, learning_rate, discount_factor, Q)
                break

            # Player O's turn
            action_O = choose_action(board,epsilon, Q)
            board[action_O] = 'O'

            if check_win(board, 'O'):
                update_q_matrix(board, action_X, board, -1, learning_rate, discount_factor, Q)
                break

        # Reset the board for the next game
        board = [' '] * 9
        
        # Testing against a random opponent
        if (epoch + 1) % epsilon_decay_interval == 0:
            agent_score = 0
            for _ in range(10):
                result = play_game(epsilon, Q, agent_turn=True)
                if result == 'X':
                    agent_score += 1
                elif result == 'Draw':
                    agent_score += 0.5
            scores.append(agent_score / 10)

    # Plotting the training progress
    plt.plot(range(epsilon_decay_interval, num_epochs + 1, epsilon_decay_interval), scores)
    plt.xlabel('Epoch')
    plt.ylabel('Average Score (against baseline)')
    plt.title('Training Progress')
    plt.show()

    print("Training and testing complete.")

if __name__ == "__main__":
    main()
