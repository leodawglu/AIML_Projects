'''
Leo Lu
CS 541 Artificial Intelligence
Summer 2023
Portland State University
Programming Assignment 2
'''
import random
import matplotlib.pyplot as plt

# Constants
BOARD_SIZE = 8

# Generate a random board configuration
def generate_random_board():
    return [random.randint(0, BOARD_SIZE - 1) for _ in range(BOARD_SIZE)]

# Calculate the number of attacking pairs for a queen
def calculate_attacking_pairs(board, row, col):
    attacking_pairs = 0
    for i in range(BOARD_SIZE):
        if i != row:
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                attacking_pairs += 1
    return attacking_pairs

# Calculate the maximum attacking pairs for a board configuration
def calculate_max_attacking_pairs(board):
    max_attacking_pairs = 0
    for row in range(BOARD_SIZE):
        max_attacking_pairs = max(max_attacking_pairs, calculate_attacking_pairs(board, row, board[row]))
    return max_attacking_pairs

# Calculate fitness of a board configuration
'''
def calculate_fitness(board):
    max_attacking_pairs = calculate_max_attacking_pairs(board)
    return 1 / (1 + max_attacking_pairs)
    '''

# Calculate fitness of a board configuration
def calculate_fitness(board):
    non_attacking_pairs = 0
    for i in range(BOARD_SIZE):
        for j in range(i + 1, BOARD_SIZE):
            if board[i] != board[j] and abs(board[i] - board[j]) != abs(i - j):
                non_attacking_pairs += 1
    return non_attacking_pairs/28


# Print the board configuration in a 2D format
def print_board(board):
    for row in board:
        print(' '.join('Q' if col == row else '.' for col in range(BOARD_SIZE)))

# Crossover two parent boards to produce two child boards
def crossover(parent1, parent2):
    crossover_point = random.randint(1, BOARD_SIZE - 2)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Apply mutation to a board configuration
def mutate(board, mutation_rate):
    if random.random() < mutation_rate:
        index = random.randint(0, BOARD_SIZE - 1)
        board[index] = random.randint(0, BOARD_SIZE - 1)

def plot_fitness_statistics(best_fitness_data, average_fitness_data, num_iterations, mutation_rate, population_size):
    generation_numbers = list(range(1, num_iterations + 1))

    #plt.plot(generation_numbers, best_fitness_data, label='Best Fitness')
    #plt.plot(generation_numbers, average_fitness_data, label='Average Fitness')
    plt.plot(generation_numbers, best_fitness_data, label=f'Best Fitness (Pop: {population_size}, Mut: {mutation_rate}, Iter: {num_iterations})')
    plt.plot(generation_numbers, average_fitness_data, label=f'Average Fitness (Pop: {population_size}, Mut: {mutation_rate}, Iter: {num_iterations})')
    plt.xlabel('Generation Number')
    plt.ylabel('Fitness')
    plt.title('Fitness Over Generations')
    plt.legend()
    # Set y-axis limits to ensure it shows 0.0 to 1.0
    plt.ylim(0.7, 1.0)
    # Add text annotations for input parameters
    #plt.text(num_iterations + 5, 0.8, f'Population Size: {population_size}')
    #plt.text(num_iterations + 5, 0.7, f'Mutation Rate: {mutation_rate}')
    #plt.text(num_iterations + 5, 0.6, f'Num Iterations: {num_iterations}')
    plt.show()


# Genetic algorithm main function
def genetic_algorithm(population_size, mutation_rate, num_iterations):
    best_fitness_per_generation = []
    best_boards_per_generation = []
    average_fitness_per_generation = []
    solution_found_iteration = -1
    num_solutions_found = 0
    population = [generate_random_board() for _ in range(population_size)]
    for iteration in range(num_iterations):
        print("Current iteration:",iteration+1)
        
        population = sorted(population, key=lambda x: calculate_fitness(x), reverse=True)
        fitness_values = [calculate_fitness(individual) for individual in population]
        average_fitness = sum(fitness_values) / population_size
        average_fitness_per_generation.append(average_fitness)
        print("Average Fitness: ", average_fitness)
        #best_fitness_index = fitness_values.index(max(fitness_values))
        best_fitness = calculate_fitness(population[0])
        best_fitness_per_generation.append(best_fitness)
        best_boards_per_generation.append(population[0])
        print("Best Fitness: ",  best_fitness)
        print_board(population[0])
        if best_fitness == 1:
            num_solutions_found += 1
            if solution_found_iteration== -1:
                solution_found_iteration = iteration
            #return population[0]
        
        # Calculate selection probabilities
        selection_probabilities = [fitness / sum(fitness_values) for fitness in fitness_values]
        
        new_population = []
        for _ in range(population_size // 2):
            #parent1 = population[random.randint(0, POPULATION_SIZE // 2 - 1)]
            #parent2 = population[random.randint(0, POPULATION_SIZE // 2 - 1)]
            # Select parents based on selection probabilities
            parent1 = random.choices(population, weights=selection_probabilities)[0]
            parent2 = random.choices(population, weights=selection_probabilities)[0]
            
            child1, child2 = crossover(parent1, parent2)
            mutate(child1,mutation_rate)
            mutate(child2,mutation_rate)
            new_population.extend([child1, child2])
        
        population = new_population

    if solution_found_iteration ==-1:
        print("Solution not found!")
    else:
        print("Solution first found at iteration: ", solution_found_iteration)
        print("Total iterations with solutions found: ", num_solutions_found)
    
    print("Average Fitness in First Generation: " , average_fitness_per_generation[0])
    print("Best Fitness in First Generation: " , best_fitness_per_generation[0])
    print("Board of Best Fitness in First Generation")
    print_board(best_boards_per_generation[0])
    # Plot fitness statistics
    plot_fitness_statistics(best_fitness_per_generation, average_fitness_per_generation, num_iterations, mutation_rate, population_size)

    return best_fitness_per_generation

def experiment_pop_1000():
    POPULATION_SIZE = 1000
    MUTATION_RATE = 0.1
    NUM_ITERATIONS = 1000
    genetic_algorithm(POPULATION_SIZE, MUTATION_RATE,NUM_ITERATIONS)

def experiment_pop_500():
    POPULATION_SIZE = 500
    MUTATION_RATE = 0.1
    NUM_ITERATIONS = 1000
    genetic_algorithm(POPULATION_SIZE, MUTATION_RATE,NUM_ITERATIONS)
    
def experiment_pop_100():
    POPULATION_SIZE = 100
    MUTATION_RATE = 0.1
    NUM_ITERATIONS = 1000
    genetic_algorithm(POPULATION_SIZE, MUTATION_RATE,NUM_ITERATIONS)
    
def experiment_mut_30():
    POPULATION_SIZE = 1000
    MUTATION_RATE = 0.3
    NUM_ITERATIONS = 1000
    genetic_algorithm(POPULATION_SIZE, MUTATION_RATE,NUM_ITERATIONS)

def experiment_mut_50():
    POPULATION_SIZE = 1000
    MUTATION_RATE = 0.5
    NUM_ITERATIONS = 1000
    genetic_algorithm(POPULATION_SIZE, MUTATION_RATE,NUM_ITERATIONS)

def experiment_mut_70():
    POPULATION_SIZE = 1000
    MUTATION_RATE = 0.7
    NUM_ITERATIONS = 1000
    genetic_algorithm(POPULATION_SIZE, MUTATION_RATE,NUM_ITERATIONS)

def main():
    experiment_mut_30()
    experiment_mut_50()
    experiment_mut_70()
    experiment_pop_100()
    experiment_pop_500()
    experiment_pop_1000()
    

if __name__ == "__main__":
    main()
