import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = np.loadtxt('545_cluster_dataset.txt')

def k_means(data, k, r=10, max_iters=100):
    best_centers = None
    best_labels = None
    best_error = float('inf')
    best_iteration_plots = []
    
    for run in range(r):
        # Initialize cluster centers randomly from data points
        initial_centers = data[np.random.choice(len(data), k, replace=False)]
        
        centers = initial_centers.copy()
        prev_centers = np.zeros_like(centers)
        labels = np.zeros(len(data), dtype=int)
        
        iteration_plots = []
        
        for iteration in range(max_iters):
            # Assign data points to the nearest cluster
            for i, point in enumerate(data):
                labels[i] = np.argmin(np.linalg.norm(centers - point, axis=1))
            
            # Update cluster centers
            for i in range(k):
                cluster_points = data[labels == i]
                if len(cluster_points) > 0:
                    centers[i] = np.mean(cluster_points, axis=0)
            
            # Check for convergence
            if np.allclose(centers, prev_centers):
                break
            
            prev_centers = centers.copy()
            
            # Plot the current iteration
            iteration_plots.append((centers.copy(), labels.copy()))
        
        # Calculate sum of squares error for this run
        error = np.sum((data - centers[labels])**2)
        
        # Update best solution if necessary
        if error < best_error:
            best_error = error
            best_centers = centers
            best_labels = labels
            best_iteration_plots = iteration_plots
    
    # Display the plots for the iteration with the lowest error
    for iteration, (centers, labels) in enumerate(best_iteration_plots):
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100)
        plt.title(f"K-Means (K = {k}) - Iteration {iteration + 1}")
        plt.show()
    
    return best_centers, best_labels, best_error

# Try different values of K
k_values = [2,3,4,7]
for k in k_values:
    centers, labels, error = k_means(data, k)
    print(f"K = {k}, Sum of Squares Error = {error:.4f}")
