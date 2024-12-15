import streamlit as st
from src.dye_utils import load_dyes, optimize_dye_proportions
from src.image_processing import read_and_preprocess_image, extract_colors_with_codes
from src.genetic_algorithm import genetic_algorithm
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load dye data (adjust path as needed)
DYE_FILE_PATH = "data/dye_detail.json"

# Title and description
st.title("Dye Proportion Optimizer")
st.markdown(
    """
    **Upload an image** to extract its dominant colors, and find optimal dye proportions 
    to replicate the target colors using a genetic algorithm.
    """
)

# Sidebar for user configuration
st.sidebar.header("Configuration")
population_size = st.sidebar.slider("Population Size", 50, 500, 100, step=50)
num_generations = st.sidebar.slider("Number of Generations", 100, 1000, 500, step=100)
num_colors = st.sidebar.slider("Number of Colors to Extract", 1, 10, 5)

# File uploader
uploaded_file = st.file_uploader("Upload an image (PNG or JPG)", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # Read and preprocess the uploaded image
    image = Image.open(uploaded_file)
    image_path = os.path.join("temp_image.jpg")
    image.save(image_path)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract dominant colors
    st.subheader("Extracting Dominant Colors")
    try:
        preprocessed_image = read_and_preprocess_image(image_path)
        dominant_colors = extract_colors_with_codes(image_path, num_colors)

        st.write("### Dominant Colors:")
        color_labels = []
        fig, ax = plt.subplots(1, num_colors, figsize=(15, 5))

        for i, (rgb, hex_code) in enumerate(dominant_colors):
            color_labels.append(f"Color {i+1}: {hex_code}")
            ax[i].imshow([[rgb]])
            ax[i].axis("off")

        st.pyplot(fig)
        st.write(color_labels)

        # Select target color for dye matching
        st.subheader("Select Target Color")
        color_options = [f"Color {i+1}: {hex_code}" for i, (_, hex_code) in enumerate(dominant_colors)]
        selected_color = st.selectbox("Choose a color to optimize dyes for:", color_options)
        target_rgb = dominant_colors[color_options.index(selected_color)][0]

        # Load dye data
        st.subheader("Loading Dye Data")
        dyes = load_dyes(DYE_FILE_PATH)
        st.write(f"Loaded {len(dyes)} dyes from `{DYE_FILE_PATH}`.")

        # Optimize dye proportions
        st.subheader("Optimizing Dye Proportions")
        dyes_with_proportions, fitness = genetic_algorithm(
            dyes, target_rgb, population_size, num_generations
        )

        # Display optimized proportions
        st.write("### Optimized Dye Proportions:")
        for dye, proportion in dyes_with_proportions.items():
            st.write(f"- {dye}: {proportion * 100:.2f}%")

        # Display final fitness
        st.write(f"**Final Fitness (Error):** {fitness:.4f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an image to get started.")
