'''
Leo Lu
CS 545 Machine Learning
Summer 2023
Portland State University
Programming Assignment 2
'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

# Load your dataset and separate features and labels
# Load the dataset from file
data = np.loadtxt("spambase.data", delimiter=",")
labels = data[:, -1]
features = data[:, :-1]


# Define class ratios
spam_ratio = 0.4  # Desired spam ratio in the final datasets
not_spam_ratio = 0.6  # Desired not-spam ratio in the final datasets

# Total instances in each set
total_instances = 4601
train_instances = 2300
test_instances = total_instances - train_instances

# Calculate instance counts for each class
train_spam_count = int(train_instances * spam_ratio)
train_not_spam_count = train_instances - train_spam_count
test_spam_count = int(test_instances * spam_ratio)
test_not_spam_count = test_instances - test_spam_count

# Split the data into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=train_instances, test_size=test_instances, stratify=labels, random_state=42)

# Further split training and test sets based on class ratios
train_spam_indices = np.where(train_labels == 1)[0]
train_not_spam_indices = np.where(train_labels == 0)[0]
test_spam_indices = np.where(test_labels == 1)[0]
test_not_spam_indices = np.where(test_labels == 0)[0]

# Select a subset of instances based on desired class ratios
selected_train_spam_indices = np.random.choice(train_spam_indices, size=min(train_spam_count, len(train_spam_indices)), replace=False)
selected_train_not_spam_indices = np.random.choice(train_not_spam_indices, size=min(train_not_spam_count, len(train_not_spam_indices)), replace=False)
selected_test_spam_indices = np.random.choice(test_spam_indices, size=min(test_spam_count, len(test_spam_indices)), replace=False)
selected_test_not_spam_indices = np.random.choice(test_not_spam_indices, size=min(test_not_spam_count, len(test_not_spam_indices)), replace=False)


# Create final datasets
final_train_indices = np.concatenate((selected_train_spam_indices, selected_train_not_spam_indices))
final_test_indices = np.concatenate((selected_test_spam_indices, selected_test_not_spam_indices))

final_train_data = train_data[final_train_indices]
final_train_labels = train_labels[final_train_indices]
final_test_data = test_data[final_test_indices]
final_test_labels = test_labels[final_test_indices]

# Compute prior probabilities for each class
total_train_instances = len(final_train_labels)
p_spam = np.sum(final_train_labels == 1) / total_train_instances
p_not_spam = np.sum(final_train_labels == 0) / total_train_instances

# Initialize dictionaries to store mean and standard deviation for each feature
mean_spam = {}
std_spam = {}
mean_not_spam = {}
std_not_spam = {}

# Calculate mean and standard deviation for each feature in both classes
for feature in range(final_train_data.shape[1]):
    mean_spam[feature] = np.mean(final_train_data[final_train_labels == 1, feature])
    std_spam[feature] = np.std(final_train_data[final_train_labels == 1, feature])
    
    mean_not_spam[feature] = np.mean(final_train_data[final_train_labels == 0, feature])
    std_not_spam[feature] = np.std(final_train_data[final_train_labels == 0, feature])
    
    # Handle zero standard deviation by assigning a minimal value
    if std_spam[feature] == 0:
        std_spam[feature] = 0.0001
    if std_not_spam[feature] == 0:
        std_not_spam[feature] = 0.0001

# Calculate class-conditional probability with smoothing
def calculate_class_conditional_prob(x, mean, std):
    exponent = np.exp(-(np.power(x - mean, 2) / (2 * np.power(std, 2))))
    prob = (1 / (np.sqrt(2 * np.pi) * std + 1e-9)) * exponent
    return prob

epsilon = 1e-9  # Small positive constant

# Function to calculate Gaussian Naïve Bayes probability for a single instance
def calculate_naive_bayes_probability(instance, mean_class, std_class, prior_class):
    log_prob = np.log(prior_class)
    for feature in range(len(instance)):
        exponent = np.exp(-(np.power(instance[feature] - mean_class[feature], 2) / (2 * np.power(std_class[feature], 2))))
        log_prob += np.log((1 / (np.sqrt(2 * np.pi) * std_class[feature])) * exponent + epsilon)
    return log_prob



# Initialize arrays to store predictions and actual labels
predictions = []
actual_labels = final_test_labels

# Iterate through each instance in the test data
for i in range(len(final_test_data)):
    log_prob_spam = calculate_naive_bayes_probability(final_test_data[i], mean_spam, std_spam, p_spam)
    log_prob_not_spam = calculate_naive_bayes_probability(final_test_data[i], mean_not_spam, std_not_spam, p_not_spam)
    
    # Classify based on the class with higher log probability
    if log_prob_spam > log_prob_not_spam:
        predictions.append(1)  # Predicted as spam
    else:
        predictions.append(0)  # Predicted as not-spam

# ... (previous code for splitting, computing mean and std, and defining the calculate_naive_bayes_probability function)

# Initialize arrays to store predictions and actual labels
predictions = []
actual_labels = final_test_labels

# Initialize variables to keep track of true positives, false positives, true negatives, false negatives
tp = fp = tn = fn = 0

# Iterate through each instance in the test data
for i in range(len(final_test_data)):
    log_prob_spam = calculate_naive_bayes_probability(final_test_data[i], mean_spam, std_spam, p_spam)
    log_prob_not_spam = calculate_naive_bayes_probability(final_test_data[i], mean_not_spam, std_not_spam, p_not_spam)
    
    # Classify based on the class with higher log probability
    if log_prob_spam > log_prob_not_spam:
        predictions.append(1)  # Predicted as spam
    else:
        predictions.append(0)  # Predicted as not-spam
        
    if predictions[i] == 1:
        if actual_labels[i] == 1:
            tp += 1
        else:
            fp += 1
    else:
        if actual_labels[i] == 0:
            tn += 1
        else:
            fn += 1

# Convert predictions to a numpy array
predictions = np.array(predictions)

# Calculate confusion matrix
conf_matrix = confusion_matrix(actual_labels, predictions)

# Plot confusion matrix
class_names = ["Not Spam", "Spam"]
fig, ax = plot_confusion_matrix(conf_mat=conf_matrix,
                                class_names=class_names,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Calculate accuracy
accuracy = np.sum(predictions == actual_labels) / len(actual_labels)
print("Accuracy:", accuracy)

# Calculate precision
precision = tp / (tp + fp)
print("Precision:", precision)

# Calculate recall
recall = tp / (tp + fn)
print("Recall:", recall)

