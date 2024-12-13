import numpy as np
from scipy.optimize import minimize

# Function to compute the error between mixed color and target color
def color_error(proportions, dyes, target_rgb):
    # Ensure the proportions sum to 1
    proportions = np.clip(proportions, 0, 1)  # Clip values to avoid negative proportions
    proportions /= proportions.sum()  # Normalize to ensure sum equals 1

    # Compute the mixed RGB value
    mixed_rgb = np.dot(proportions, dyes)

    # Calculate the Euclidean distance between the mixed color and target color
    error = np.linalg.norm(mixed_rgb - target_rgb)
    
    return error

def mix_dyes(dyes, target_rgb):
    # Initial guess for the proportions (uniform distribution)
    initial_proportions = np.ones(len(dyes)) / len(dyes)
    
    # Bounds for the proportions (all must be between 0 and 1)
    bounds = [(0, 1)] * len(dyes)
    
    # Optimization to minimize color error
    result = minimize(color_error, initial_proportions, args=(dyes, target_rgb), bounds=bounds, method='SLSQP')
    
    # Return the optimized proportions and the resulting color
    optimized_proportions = result.x / result.x.sum()  # Normalize to ensure sum equals 1
    mixed_color = np.dot(optimized_proportions, dyes)
    
    return optimized_proportions, mixed_color

# Example usage

# RGB values of dyes (as an example, replace with actual dye RGB values)
dyes = np.array([
    [255, 0, 0],   # Red
    [0, 255, 0],   # Green
    [0, 0, 255],   # Blue
    [255, 255, 0], # Yellow
    [255, 165, 0]  # Orange
])

# Target RGB value (replace with the target you want to match)
target_rgb = np.array([200, 100, 50])

# Find the best dye proportions to match the target color
proportions, mixed_color = mix_dyes(dyes, target_rgb)

print("Optimized Dye Proportions:", proportions)
print("Mixed Color (RGB):", mixed_color)
