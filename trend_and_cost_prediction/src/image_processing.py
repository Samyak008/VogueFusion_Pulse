import numpy as np
import random
import json
from collections import Counter
from sklearn.cluster import KMeans
from matplotlib.colors import to_hex, to_rgb
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cv2

def read_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Extract colors from image
def extract_colors_with_codes(image_path, num_colors):
    image = read_and_preprocess_image(image_path)
    plt.imshow(image)
    plt.title("Saree Image")
    plt.axis("off")
    plt.show()
    pixels = image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    counts = Counter(labels)
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    sorted_colors = [colors[item[0]] for item in sorted_items]
    sorted_counts = [item[1] for item in sorted_items]

    hex_codes = [to_hex(np.array(color) / 255) for color in sorted_colors]

    return sorted_colors, hex_codes, sorted_counts

# Display RGB and HEX codes
def display_color_codes(colors, hex_codes, counts):
    total_pixels = sum(counts)
    print("\nDominant Colors and HEX Codes:")
    for color, hex_code, count in zip(colors, hex_codes, counts):
        proportion = count / total_pixels * 100
        print(f"RGB: {color}, HEX: {hex_code}, Proportion: {proportion:.2f}%")

# Plot color proportions
def plot_colors(colors, counts):
    total_pixels = sum(counts)
    proportions = [count / total_pixels for count in counts]

    plt.figure(figsize=(8, 4))
    for i, (color, proportion) in enumerate(zip(colors, proportions)):
        plt.bar(i, proportion, color=np.array(color) / 255, edgecolor="black")

    plt.xticks(range(len(colors)), [f"Color {i+1}" for i in range(len(colors))])
    plt.ylabel("Proportion")
    plt.title("Color Proportions in Saree")
    plt.show()