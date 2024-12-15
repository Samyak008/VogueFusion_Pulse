import numpy as np
import random
import json
from collections import Counter
from sklearn.cluster import KMeans
from matplotlib.colors import to_hex, to_rgb
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cv2

with open("data/dye_detail.json", "r") as file:
    dyes = json.load(file)

# Extract RGB values from dyes
rgb_values = np.array([dye["rgb"] for dye in dyes])

# Genetic Algorithm Parameters
POPULATION_SIZE = 100
NUM_GENERATIONS = 500
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 25

def fitness_function(proportions, target_rgb):
    mixed_rgb = np.dot(proportions, rgb_values)
    error = np.sqrt(np.sum((mixed_rgb - target_rgb) ** 2))  # Euclidean distance
    return error

# Create an initial population
def create_population(size, num_dyes):
    return np.random.dirichlet(np.ones(num_dyes), size)

# Select parents using tournament selection
def tournament_selection(population, fitness, tournament_size):
    selected_indices = random.sample(range(len(population)), tournament_size)
    best_index = min(selected_indices, key=lambda idx: fitness[idx])
    return population[best_index]

# Perform crossover between two parents
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2
    return parent1, parent2

# Perform mutation on a single individual
def mutate(individual):
    if random.random() < MUTATION_RATE:
        index = random.randint(0, len(individual) - 1)
        adjustment = np.random.uniform(-0.1, 0.1)
        individual[index] = max(0, min(1, individual[index] + adjustment))
    return individual / individual.sum()  # Ensure proportions sum to 1

# Main Genetic Algorithm for a single target color
def genetic_algorithm(target_rgb):
    num_dyes = len(dyes)
    population = create_population(POPULATION_SIZE, num_dyes)
    best_solution = None
    best_fitness = float("inf")

    for generation in range(NUM_GENERATIONS):
        fitness = np.array([fitness_function(individual, target_rgb) for individual in population])
        next_population = []

        for _ in range(POPULATION_SIZE // 2):
            # Select parents
            parent1 = tournament_selection(population, fitness, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitness, TOURNAMENT_SIZE)

            # Crossover and mutation
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)

            next_population.extend([child1, child2])

        population = np.array(next_population)

        # Track the best solution
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = population[current_best_idx]

    return best_solution, best_fitness
