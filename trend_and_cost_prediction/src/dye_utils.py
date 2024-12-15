import numpy as np
import random
import json
from collections import Counter
from sklearn.cluster import KMeans
from matplotlib.colors import to_hex, to_rgb
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cv2

def load_dyes(file_path):
    with open(file_path, "r") as file:
        return json.load(file)
    
def optimize_dye_proportions(dyes_with_proportions, target_rgb, threshold=1e-3):
    """
    Optimizes the proportions of dyes to match a target RGB color.
    
    Args:
        dyes_with_proportions (list): A list of dictionaries, each with keys 'name' and 'rgb'.
        target_rgb (list or np.array): The target RGB color as a list or array.
        threshold (float): Minimum proportion threshold to include a dye in the result.
    
    Returns:
        dict: A dictionary with the following keys:
              - 'necessary_dyes': List of tuples (dye name, proportion) for dyes with non-negligible proportions.
              - 'resulting_rgb': The RGB color obtained from the optimized proportions.
              - 'target_rgb': The target RGB color.
    """
    target_rgb = np.array(target_rgb)
    dye_rgbs = np.array([dye["rgb"] for dye in dyes_with_proportions])
    
    # Define the objective function (mean squared error)
    def objective(proportions):
        resulting_rgb = np.dot(proportions, dye_rgbs)
        error = np.sum((target_rgb - resulting_rgb) ** 2)
        return error

    # Constraints: proportions must sum to 1 and be non-negative
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = [(0, 1) for _ in range(len(dyes_with_proportions))]
    
    # Initial guess (equal proportions)
    initial_guess = np.ones(len(dyes_with_proportions)) / len(dyes_with_proportions)
    
    # Solve the optimization problem
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
    proportions = result.x
    
    # Filter out dyes with negligible proportions
    necessary_dyes = [
        (dyes_with_proportions[i]['name'], proportions[i])
        for i in range(len(dyes_with_proportions))
        if proportions[i] > threshold
    ]
    
    # Calculate resulting RGB
    resulting_rgb = np.dot(proportions, dye_rgbs)
    
    return {
        "necessary_dyes": necessary_dyes,
        "resulting_rgb": resulting_rgb,
        "target_rgb": target_rgb
    }