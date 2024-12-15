# import streamlit as st
# import numpy as np
# import random
# import json

# # Constants for genetic algorithm
# POPULATION_SIZE = 100
# NUM_GENERATIONS = 500
# MUTATION_RATE = 0.1
# CROSSOVER_RATE = 0.7
# TOURNAMENT_SIZE = 25

# # Load dye details from JSON
# with open("data/dye_detail.json", "r") as file:
#     dyes = json.load(file)

# # Extract RGB values from dyes
# rgb_values = np.array([dye["rgb"] for dye in dyes])


# def normalize_proportions(proportions):
#     total_proportion = sum(proportions)
#     # Normalize the proportions so they sum to 100% (i.e., 1 in the [0,1] range)
#     return [proportion / total_proportion for proportion in proportions]


# def select_majority_colors_and_adjust(proportions, dyes, threshold=5):
#     # Sort dyes by proportion in descending order
#     sorted_indices = np.argsort(proportions)[::-1]
    
#     # Select the majority colors (above a certain threshold, e.g., 5%)
#     selected_dyes = []
#     selected_proportions = []
    
#     for idx in sorted_indices:
#         if proportions[idx] >= threshold / 100:  # Ensure threshold is in the 0-1 scale
#             selected_dyes.append(dyes[idx])
#             selected_proportions.append(proportions[idx])

#     # Normalize the selected proportions to make sure they sum to 1
#     if selected_proportions:
#         adjusted_proportions = normalize_proportions(selected_proportions)
#     else:
#         # If no dyes pass the threshold, return empty list (shouldn't happen often)
#         adjusted_proportions = []

#     return selected_dyes, adjusted_proportions

# # Fitness function
# def fitness_function(proportions, target_rgb, rgb_values):
#     mixed_rgb = np.dot(proportions, rgb_values)
#     error = np.sqrt(np.sum((mixed_rgb - target_rgb) ** 2))  # Euclidean distance
#     return error

# # Create an initial population
# def create_population(size, num_dyes):
#     return np.random.dirichlet(np.ones(num_dyes), size)

# # Select parents using tournament selection
# def tournament_selection(population, fitness, tournament_size):
#     selected_indices = random.sample(range(len(population)), tournament_size)
#     best_index = min(selected_indices, key=lambda idx: fitness[idx])
#     return population[best_index]

# # Perform crossover between two parents
# def crossover(parent1, parent2):
#     if random.random() < CROSSOVER_RATE:
#         crossover_point = random.randint(1, len(parent1) - 1)
#         child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
#         child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
#         return child1, child2
#     return parent1, parent2

# # Perform mutation on a single individual
# def mutate(individual):
#     if random.random() < MUTATION_RATE:
#         index = random.randint(0, len(individual) - 1)
#         adjustment = np.random.uniform(-0.1, 0.1)
#         individual[index] = max(0, min(1, individual[index] + adjustment))
#     return individual / individual.sum()  # Ensure proportions sum to 1

# # Genetic algorithm for a single target color
# def genetic_algorithm(target_rgb):
#     num_dyes = len(dyes)
#     population = create_population(POPULATION_SIZE, num_dyes)
#     best_solution = None
#     best_fitness = float("inf")

#     for generation in range(NUM_GENERATIONS):
#         fitness = np.array([fitness_function(individual, target_rgb, rgb_values) for individual in population])
#         next_population = []

#         for _ in range(POPULATION_SIZE // 2):
#             parent1 = tournament_selection(population, fitness, TOURNAMENT_SIZE)
#             parent2 = tournament_selection(population, fitness, TOURNAMENT_SIZE)

#             child1, child2 = crossover(parent1, parent2)
#             child1 = mutate(child1)
#             child2 = mutate(child2)

#             next_population.extend([child1, child2])

#         population = np.array(next_population)

#         current_best_idx = np.argmin(fitness)
#         if fitness[current_best_idx] < best_fitness:
#             best_fitness = fitness[current_best_idx]
#             best_solution = population[current_best_idx]
#     selected_dyes, adjusted_proportions = select_majority_colors_and_adjust(best_solution, dyes)
#     return adjusted_proportions, best_fitness,selected_dyes

# # Streamlit app
# def main():
#     st.title("Saree Dye Optimization using Genetic Algorithm")

#     if st.button("Optimize All Colors"):
#         if "colors" in st.session_state and "hex_codes" in st.session_state and "counts" in st.session_state:
#             colors = st.session_state["colors"]
#             hex_codes = st.session_state["hex_codes"]

#             # Display extracted colors
#             st.write("### Extracted Colors")
#             for i, (color, hex_code) in enumerate(zip(colors, hex_codes)):
#                 st.markdown(f"**Color {i + 1}:** RGB {color}, HEX {hex_code}")

#             # Perform dye optimization for all colors
#             st.write("### Dye Optimization Results")
#             for i, target_rgb in enumerate(colors):
#                 st.write(f"#### Color {i + 1}")
#                 best_proportions, best_error, selected_dyes = genetic_algorithm(target_rgb)
#                 print(best_proportions)
                
#                 resulting_rgb = np.dot(best_proportions, [dye["rgb"] for dye in selected_dyes]).round().astype(int)

#                 # resulting_rgb = np.dot(best_proportions, [dye["rgb"] for dye in dyes]).round().astype(int)

#                 st.write("**Target RGB:**", target_rgb)
#                 st.write("**Resulting RGB:**", resulting_rgb)
#                 st.write("**Optimal Dye Proportions:**")

#                 for dye, proportion in zip(dyes, best_proportions):
#                     if proportion > 0.01:  # Filter negligible proportions
#                         st.write(f"{dye['color']}: {proportion:.2%}")

#                 st.write("---")

#         else:
#             st.error("No colors found. Please extract colors from the previous page first.")

# if __name__ == "__main__":
#     main()
import streamlit as st
import numpy as np
import random
import json
from PIL import Image

# Constants for genetic algorithm
POPULATION_SIZE = 100
NUM_GENERATIONS = 500
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 25

# Load dye details from JSON
with open("data/dye_detail.json", "r") as file:
    dyes = json.load(file)

# Extract RGB values from dyes
rgb_values = np.array([dye["rgb"] for dye in dyes])


def normalize_proportions(proportions):
    total_proportion = sum(proportions)
    return [proportion / total_proportion for proportion in proportions]


def select_majority_colors_and_adjust(proportions, dyes, threshold=5):
    sorted_indices = np.argsort(proportions)[::-1]
    selected_dyes = []
    selected_proportions = []
    
    for idx in sorted_indices:
        if proportions[idx] >= threshold / 100:
            selected_dyes.append(dyes[idx])
            selected_proportions.append(proportions[idx])

    if selected_proportions:
        adjusted_proportions = normalize_proportions(selected_proportions)
    else:
        adjusted_proportions = []
    
    return selected_dyes, adjusted_proportions


# Fitness function
def fitness_function(proportions, target_rgb, rgb_values):
    mixed_rgb = np.dot(proportions, rgb_values)
    error = np.sqrt(np.sum((mixed_rgb - target_rgb) ** 2))
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
    return individual / individual.sum()


# Genetic algorithm for a single target color
def genetic_algorithm(target_rgb):
    num_dyes = len(dyes)
    population = create_population(POPULATION_SIZE, num_dyes)
    best_solution = None
    best_fitness = float("inf")

    for generation in range(NUM_GENERATIONS):
        fitness = np.array([fitness_function(individual, target_rgb, rgb_values) for individual in population])
        next_population = []

        for _ in range(POPULATION_SIZE // 2):
            parent1 = tournament_selection(population, fitness, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitness, TOURNAMENT_SIZE)

            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)

            next_population.extend([child1, child2])

        population = np.array(next_population)

        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = population[current_best_idx]
    
    selected_dyes, adjusted_proportions = select_majority_colors_and_adjust(best_solution, dyes)
    return adjusted_proportions, best_fitness, selected_dyes


# Streamlit app
def main():
    st.title("Saree Dye Optimization using Genetic Algorithm")

    if st.button("Optimize All Colors"):
        if "colors" in st.session_state and "hex_codes" in st.session_state and "counts" in st.session_state:
            colors = st.session_state["colors"]
            hex_codes = st.session_state["hex_codes"]

            # Display extracted colors
            st.write("### Extracted Colors")
            for i, (color, hex_code) in enumerate(zip(colors, hex_codes)):
                # Display color preview and HEX code
                st.markdown(f"**Color {i + 1}:**")
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"<div style='background-color: rgb{tuple(color)}; width: 50px; height: 50px; border-radius: 50%;'></div>", unsafe_allow_html=True)
                with col2:
                    st.write(f"**RGB:** {color}")
                    st.write(f"**HEX:** {hex_code}")

            st.write("### Dye Optimization Results")
            for i, target_rgb in enumerate(colors):
                st.write(f"#### Color {i + 1}")
                best_proportions, best_error, selected_dyes = genetic_algorithm(target_rgb)
                
                resulting_rgb = np.dot(best_proportions, [dye["rgb"] for dye in selected_dyes]).round().astype(int)
                st.session_state["best_proportion"] = best_proportions
                st.session_state["selected_dyes"] = selected_dyes
                st.session_state["best_error"] = best_error
                st.write(f"**Target RGB:** {target_rgb}")
                st.write(f"**Resulting RGB:** {resulting_rgb}")
                st.write("**Optimal Dye Proportions:**")

                # Display optimal dye proportions with color blocks and progress bars
                for dye, proportion in zip(dyes, best_proportions):
                    if proportion > 0.01:  # Filter negligible proportions
                        st.markdown(f"**{dye['color']}:**")
                        st.write(f"Proportion: {proportion:.2%}")
                        st.progress(proportion)  # Show progress bar for proportion
                        st.markdown(f"<div style='background-color: rgb{tuple(dye['rgb'])}; width: 100px; height: 30px;'></div>", unsafe_allow_html=True)
                        st.write("")
                
                st.write("---")
        else:
            st.error("No colors found. Please extract colors from the previous page first.")


if __name__ == "__main__":
    main()
