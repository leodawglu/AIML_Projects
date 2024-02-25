import random
import numpy as np
import matplotlib.pyplot as plt

# Define the tic-tac-toe game
class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        self.winner = None

    def is_empty(self, position):
        return self.board[position] == ' '

    def make_move(self, position):
        if self.is_empty(position):
            self.board[position] = self.current_player
            self.check_winner()

            if self.current_player == 'X':
                self.current_player = 'O'
            else:
                self.current_player = 'X'

    def check_winner(self):
        win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                          (0, 3, 6), (1, 4, 7), (2, 5, 8),
                          (0, 4, 8), (2, 4, 6)]

        for condition in win_conditions:
            a, b, c = condition
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                self.winner = self.board[a]
                return

        if ' ' not in self.board:
            self.winner = 'Draw'

    def is_game_over(self):
        return self.winner is not None

    def display(self):
        for i in range(0, 9, 3):
            print(f"{self.board[i]} | {self.board[i+1]} | {self.board[i+2]}")
            if i < 6:
                print("---------")

# Q-learning agent
class QLearningAgent:
    def __init__(self, epsilon=0.1, learning_rate=0.1, discount_factor=0.9):
        self.q_matrix = np.zeros((3 ** 9, 9))  # Q-matrix: states x actions
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def state_to_index(self, state):
        index = 0
        for i, mark in enumerate(state):
            if mark == 'X':
                index += 3 ** i
            elif mark == 'O':
                index += 2 * (3 ** i)
        return index

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([i for i, mark in enumerate(state) if mark == ' '])
        else:
            state_index = self.state_to_index(state)
            return np.argmax(self.q_matrix[state_index])

    def update_q_matrix(self, state, action, reward, next_state):
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)
        self.q_matrix[state_index, action] += self.learning_rate * (
            reward + self.discount_factor * np.max(self.q_matrix[next_state_index]) - self.q_matrix[state_index, action]
        )

# Training the Q-learning agent
def train_q_learning_agent(agent, episodes, delta, epsilon, m_decay_episodes, test_games):
    tic_tac_toe = TicTacToe()
    total_reward = 0
    min_epsilon = 0.0001
    agent_wins = 0
    agent_draws = 0
    agent_losses = 0
    divisor = 10
    partial_avg=0 #for smoothing plot
    count = 0
    average_test_scores = []
    overall_average_test_score = 0
    for episode in range(episodes):
        #epsilon = max_epsilon - episode * delta
        state = tic_tac_toe.board[:]

        while not tic_tac_toe.is_game_over():
            action = agent.choose_action(state)
            tic_tac_toe.make_move(action)

            next_state = tic_tac_toe.board[:]
            reward = 0

            if tic_tac_toe.is_game_over():
                print("Games is over!")
                if tic_tac_toe.winner == 'X':
                    print("Agent Won!")
                    reward = 1
                elif tic_tac_toe.winner == 'O':
                    print("Agent Lost!")
                    reward = 0
                else:
                    print("DRAW!")
                    reward = 0.5  # Draw

            total_reward += reward
            agent.update_q_matrix(state, action, reward, next_state)
            state = next_state

        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.1f}")
        
            
        test_score,wins, draws, losses = test_agent(agent, test_games)
        agent_wins += wins
        agent_draws += draws
        agent_losses += losses
        average_test_score = test_score/test_games
        count += 1
        partial_avg += average_test_score
        if count == divisor:
            count = 0
            partial_avg /= divisor
            average_test_scores.append(partial_avg)
            partial_avg = 0
        overall_average_test_score += average_test_score
        print(f"Testing against random opponent ({test_games} games):")
        print(f"Average score: {average_test_score} (win: +1, draw: +0.5, loss: 0)")
        
        # Decay epsilon every epsilon_decay_episodes
        if (episode + 1) % m_decay_episodes == 0:
            epsilon = epsilon - delta
            epsilon = max(epsilon - delta, min_epsilon) 
            agent.epsilon = epsilon
            
        print(f"Epsilon: {epsilon}")    
        #reset board
        tic_tac_toe = TicTacToe()
    #overall_average_test_score /= episodes
    print("Final Results for Q agent")
    print(f"Overall game results against random opponent, wins: {agent_wins}, draws: {agent_draws}, losses: {agent_losses}")
    print(f"Overall average test score: {overall_average_test_score}")
    
    # Plot the training progress
    plt.plot(range(divisor, episodes+1, divisor), average_test_scores)
    plt.xlabel("Episode")
    plt.ylabel("Average Test Score")
    plt.title("Training Progress for Q-learning Agent")
    plt.show()


# Player vs Q-learning agent
def play_games_with_agent(agent, games):
    tic_tac_toe = TicTacToe()
    agent_wins = 0
    player_wins = 0
    for _ in range(games):
        while not tic_tac_toe.is_game_over():
            tic_tac_toe.display()

            if tic_tac_toe.current_player == 'X':  # Agent's turn
                state = tic_tac_toe.board[:]
                action = agent.choose_action(state)
                print("Agent's Turn!")
                tic_tac_toe.make_move(action)
            else:  # Human player's turn
                print("Your Turn!")
                position = int(input("Enter the position (0-8): "))
                if not tic_tac_toe.is_empty(position):
                    print("Invalid move! Position is already taken.")
                    continue
                tic_tac_toe.make_move(position)

        tic_tac_toe.display()

        if tic_tac_toe.winner == 'X':
            print("Agent Won!")
            agent_wins += 1
        elif tic_tac_toe.winner == 'O':
            print("You Won!")
            player_wins += 1
        else:
            print("It's a Draw!")

        tic_tac_toe = TicTacToe()  # Reset the game board
    print(f"Player won {player_wins}/10 games.")

# Test the trained agent against a random opponent
def test_agent(agent, games):
    total_score = 0
    agent_wins = 0
    agent_draws = 0
    agent_losses = 0
    for _ in range(games):
        tic_tac_toe = TicTacToe()

        while not tic_tac_toe.is_game_over():
            if tic_tac_toe.current_player == 'X':
                state = tic_tac_toe.board[:]
                action = agent.choose_action(state)
                tic_tac_toe.make_move(action)
            else:
                empty_positions = [i for i, mark in enumerate(tic_tac_toe.board) if mark == ' ']
                random_action = random.choice(empty_positions)
                tic_tac_toe.make_move(random_action)

        if tic_tac_toe.winner == 'X':
            agent_wins+=1
            total_score += 1
        elif tic_tac_toe.winner == 'Draw':
            agent_draws+=1
            total_score += 0.5
        else:
            agent_losses+=1

    print(f"Q-Agent results: {agent_wins} wins, {agent_draws}draws, {agent_losses}losses.")
    return total_score, agent_wins, agent_draws, agent_losses

# Main program
if __name__ == "__main__":
    episodes = 5000
    m_decay_episodes = 50
    #delta = 0.00009
    delta = 0.005
    #delta = 0.00003
    start_epsilon = 0.5
    test_games = 10
    agent = QLearningAgent(epsilon=0.1, learning_rate=0.1, discount_factor=0.9)
    

    print("Training Q-learning agent...")
    train_q_learning_agent(agent, episodes, delta, start_epsilon, m_decay_episodes, test_games)
    
    print("Training complete! Let's play 10 games against the Q-learning agent as 'O'.")
    play_games_with_agent(agent, 10)