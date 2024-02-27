'''
Leo Lu
Summer 2023
CS 545 Machine Leaning
Portland State University

Program 1
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

class Perceptron:
    
    def __init__(self, hidden_units, learning_rate=0.1, epochs=50, momentum=0.9, training_percentage=100):
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.momentum = momentum
        
        self.training_data = self.get_data_from_file('noheader_train.csv',training_percentage)
        self.testing_data = self.get_data_from_file('noheader_test.csv')
        self.number_of_columns = self.training_data.shape[1] - 1
        self.input_units = self.number_of_columns - 1  # Excluding the bias input, 784 (28x28)
        
        # Initialize weights between input and hidden layers
        self.weights_input_hidden = np.random.uniform(-0.05, 0.05, (self.input_units + 1, self.hidden_units))
        # Initialize weights between hidden and output layers
        self.weights_hidden_output = np.random.uniform(-0.05, 0.05, (self.hidden_units + 1, 10))
        
        # Initialize previous weights for momentum
        self.prev_weights_hidden_output = np.zeros((self.hidden_units + 1, 10))
        self.prev_weights_input_hidden = np.zeros((self.input_units + 1, self.hidden_units))
     
        self.num_of_test_samples = self.testing_data.shape[0]
        self.num_of_training_samples = self.training_data.shape[0]
        
    def get_data_from_file(self, filename, percentage = 100):
        get_dataframe = pd.read_csv(filename, header=None)
        get_data_array_attribute = get_dataframe.values
        data_array_converted_to_float = np.asarray(get_data_array_attribute, dtype=float)
        data_array_converted_to_float[:, 1:] = data_array_converted_to_float[:, 1:] / 255
        # Add bias input (set to 1)
        rows = data_array_converted_to_float.shape[0]
        append_bias_ones_to_data_array = np.append(data_array_converted_to_float, np.ones((rows, 1)), axis=1)
        
        # Randomly shuffle the data examples to ensure that data is balanced amongst 10 classes
        np.random.shuffle(append_bias_ones_to_data_array)
        # Calculate the number of data examples to extract based on the specified percentage
        num_examples_to_extract = int(rows * (percentage / 100))

        # Get the required percentage of data examples
        data_subset = append_bias_ones_to_data_array[:num_examples_to_extract]
        return data_subset
        
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_pass(self, inputs):
        # Calculate input values for hidden layer
        hidden_layer_input = np.dot(inputs, self.weights_input_hidden)
        # Apply sigmoid activation function to hidden layer input to get hidden layer output
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        
        # Add bias hidden unit (set to 1) to hidden layer output
        hidden_layer_output = np.append(hidden_layer_output, 1)
        
        # Calculate input to output layer
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        # Apply sigmoid activation function to output layer input to get output layer output
        output_layer_output = self.sigmoid(output_layer_input)
        
        return hidden_layer_output, output_layer_output
        
    def backpropagation(self, inputs, hidden_output, output_output, targets):
        # Output layer error and delta
        output_error = targets - output_output
        output_delta = output_error * self.sigmoid_derivative(output_output)
        
        # Hidden layer error (excluding bias)
        hidden_error = np.dot(self.weights_hidden_output[:-1, :], output_delta)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output[:-1])
        
        # Update weights between hidden and output layers with momentum term
        self.weights_hidden_output[:-1, :] += self.learning_rate * np.outer(hidden_output[:-1], output_delta) + \
                                            self.momentum * (self.weights_hidden_output[:-1, :] - self.prev_weights_hidden_output[:-1, :])
        self.prev_weights_hidden_output = np.copy(self.weights_hidden_output)
        
        # Update weights between input and hidden layers with momentum term
        self.weights_input_hidden += self.learning_rate * np.outer(inputs, hidden_delta) + \
                                    self.momentum * (self.weights_input_hidden - self.prev_weights_input_hidden)
        self.prev_weights_input_hidden = np.copy(self.weights_input_hidden)




    '''
    def backpropagation(self, inputs, hidden_output, output_output, targets):
        # Output layer error and delta
        output_error = targets - output_output
        output_delta = output_error * self.sigmoid_derivative(output_output)
        
        # Hidden layer error
        hidden_error = np.dot(self.weights_hidden_output, output_delta)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)
        
        # Update weights between hidden and output layers with momentum term
        self.weights_hidden_output += self.learning_rate * np.outer(hidden_output, output_delta) + \
                                     self.momentum * (self.weights_hidden_output - self.prev_weights_hidden_output)
        self.prev_weights_hidden_output = np.copy(self.weights_hidden_output)
        
        # Update weights between input and hidden layers with momentum term
        self.weights_input_hidden += self.learning_rate * np.outer(inputs, hidden_delta[:-1]) + \
                                     self.momentum * (self.weights_input_hidden - self.prev_weights_input_hidden)
        self.prev_weights_input_hidden = np.copy(self.weights_input_hidden)
        
        
        
    def backpropagation(self, inputs, hidden_output, output_output, targets):
        # Output layer error and delta
        output_error = targets - output_output
        output_delta = output_error * self.sigmoid_derivative(output_output)
        
        # Hidden layer error
        hidden_error = np.dot(self.weights_hidden_output[:-1, :], output_delta)  # Exclude bias node weights
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output[:-1])  # Exclude bias node activation
        
        # Update weights between hidden and output layers with momentum term
        self.weights_hidden_output[:-1, :] += self.learning_rate * np.outer(hidden_output[:-1], output_delta) + \
                                            self.momentum * (self.weights_hidden_output[:-1, :] - self.prev_weights_hidden_output[:-1, :])
        self.prev_weights_hidden_output = np.copy(self.weights_hidden_output[:-1, :])
        
        # Update weights between input and hidden layers with momentum term
        self.weights_input_hidden[:, :-1] += self.learning_rate * np.outer(inputs, hidden_delta) + \
                                            self.momentum * (self.weights_input_hidden[:-1, :] - self.prev_weights_input_hidden[:-1, :])
        self.prev_weights_input_hidden = np.copy(self.weights_input_hidden[:, :-1])'''
    
    def training(self):
        training_accuracy_list = []
        testing_accuracy_list = []
        
        for epoch in range(self.epochs):
            print('Starting epoch:', epoch)
            
            for i in range(self.num_of_training_samples):
                # Extract input and target values
                inputs = self.training_data[i, 1:]
                target = np.ones(10)*0.1
                target[int(self.training_data[i, 0])] = 0.9
                
                # Forward pass
                hidden_output, output_output = self.forward_pass(inputs)
                
                # Backpropagation
                self.backpropagation(inputs, hidden_output, output_output, target)
                
            # Calculate accuracy after each epoch
            training_accuracy = self.testing(self.training_data)
            training_accuracy_list.append(training_accuracy)
            print('Training accuracy:', training_accuracy)
            
            testing_accuracy = self.testing(self.testing_data)
            testing_accuracy_list.append(testing_accuracy)  
            print('Testing accuracy:', testing_accuracy)
                
        return training_accuracy_list, testing_accuracy_list
    
    def testing(self, test_data):
        correct_predictions = 0
        
        # Iterate through each test example in the test dataset
        for i in range(test_data.shape[0]):
            # Extract input and target values
            inputs = test_data[i, 1:]
            # The true class label for the current test example
            target_class = int(test_data[i, 0])
            
            # Forward pass through the network to get output
            _, output_activation_value = self.forward_pass(inputs)
            # Predict the class with the highest activation value (highest probability)
            predicted_class = np.argmax(output_activation_value)
            
            #If predicted class matches target class, increment counter
            if predicted_class == target_class:
                correct_predictions += 1
                
        accuracy = correct_predictions / test_data.shape[0]
        return accuracy
    
    def plot(self, training_accuracy_list, testing_accuracy_list, experiment_description):
        plt.plot(range(self.epochs), training_accuracy_list, label="Training Accuracy")
        plt.plot(range(self.epochs), testing_accuracy_list, label="Test Accuracy")
        
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{experiment_description}")
        plt.legend()
        plt.show()
        
    def plot_exp2(self, training_accuracy_list, testing_accuracy_list, experiment_description, exp1_training, exp1_testing):
        plt.plot(range(self.epochs), training_accuracy_list, label="Training Accuracy")
        plt.plot(range(self.epochs), testing_accuracy_list, label="Test Accuracy")
        plt.plot(range(self.epochs), exp1_training, label="Exp 1 Training Accuracy")
        plt.plot(range(self.epochs), exp1_testing, label="Exp 1 Test Accuracy")
        
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{experiment_description}")
        plt.legend()
        plt.show()
        
    def confusion_matrix(self, experiment_description):
        matrix = np.zeros((10, 10), int)
        vals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        for i in range(self.num_of_test_samples):
            inputs = self.testing_data[i, 1:]
            target_class = int(self.testing_data[i, 0])
            
            _, output_output = self.forward_pass(inputs)
            predicted_class = np.argmax(output_output)
            
            matrix[target_class, predicted_class] += 1

        fig, ax = plot_confusion_matrix(
            conf_mat=matrix, class_names=vals, colorbar=True, fontcolor_threshold=1, cmap="coolwarm"
        )
        plt.title(f"{experiment_description}")
        plt.show()
        

def experiment1():
    hidden_units_list = [20, 50, 100]
    learning_rate = 0.1
    #epochs = 3
    epochs = 50

    training_accuracy_lists = []
    testing_accuracy_lists = []

    for hidden_units in hidden_units_list:
        current_model = Perceptron(hidden_units, learning_rate, epochs)
        training_accuracy_list, testing_accuracy_list = current_model.training()
        training_accuracy_lists.append(training_accuracy_list)
        testing_accuracy_lists.append(testing_accuracy_list)
        current_model.plot(training_accuracy_list, testing_accuracy_list, f"Exp 1 - Hidden Units = {hidden_units}")
        current_model.confusion_matrix(f"Exp 1 - Hidden Units = {hidden_units}")

    return training_accuracy_lists, testing_accuracy_lists


def experiment2(exp1_training_list,exp1_test_list):
    momentum_values = [0, 0.25, 0.5]
    hidden_units = 100
    learning_rate = 0.1
    #epochs = 3
    epochs = 50

    training_accuracy_lists = []
    testing_accuracy_lists = []

    for momentum in momentum_values:
        current_model = Perceptron(hidden_units, learning_rate, epochs, momentum)
        training_accuracy_list, testing_accuracy_list = current_model.training()
        training_accuracy_lists.append(training_accuracy_list)
        testing_accuracy_lists.append(testing_accuracy_list)
        current_model.plot_exp2(training_accuracy_list, testing_accuracy_list, f"Exp 2 - Momentum = {momentum}", exp1_training_list,exp1_test_list)
        current_model.confusion_matrix(f"Exp 2 - Momentum = {momentum}")

    return training_accuracy_lists, testing_accuracy_lists


def experiment3():
    hidden_units = 100
    learning_rate = 0.1
    epochs = 50
    momentum = 0.9

    training_percentages = [25, 50]

    training_accuracy_lists = []
    testing_accuracy_lists = []

    for percentage in training_percentages:
        current_model = Perceptron(hidden_units, learning_rate, epochs, momentum, percentage)
        training_accuracy_list, testing_accuracy_list = current_model.training()
        training_accuracy_lists.append(training_accuracy_list)
        testing_accuracy_lists.append(testing_accuracy_list)
        current_model.plot(training_accuracy_list, testing_accuracy_list, f"Exp 3 - Percentage = {percentage}%")
        current_model.confusion_matrix(f"Exp 3 - Percentage = {percentage}%")

    return training_accuracy_lists, testing_accuracy_lists
        
def main():
    training_accuracy_lists_exp1, testing_accuracy_lists_exp1 = experiment1()
    experiment2(training_accuracy_lists_exp1[-1], testing_accuracy_lists_exp1[-1])
    experiment3()


if __name__ == "__main__":
    main()
