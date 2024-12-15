# streamlit_app.py
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from src.image_processing import (
    extract_colors_with_codes,
    display_color_codes,
    plot_colors,
)
import os

def main():
    st.title("Saree Design Color Extraction and Dye Optimization")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image of the saree:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Saree Image", use_container_width=True)

        # Save the image temporarily
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)

        # Number of colors to extract
        num_colors = st.slider("Select number of dominant colors to extract:", min_value=1, max_value=10, value=5)

        if st.button("Extract Colors"):
            try:
                # Extract colors and display them
                colors, hex_codes, counts = extract_colors_with_codes(temp_image_path, num_colors)
                st.session_state["colors"] = colors
                st.session_state["hex_codes"] = hex_codes
                st.session_state["counts"] = counts

                print(colors,counts)
                # Display proportions
                total_pixels = sum(counts)
                proportions = [(count / total_pixels) * 100 for count in counts]

                # Show the colors as a bar chart
                fig, ax = plt.subplots(figsize=(8, 4))
                for i, (color, proportion) in enumerate(zip(colors, proportions)):
                    ax.bar(i, proportion, color=[c / 255 for c in color], edgecolor="black")
                ax.set_xticks(range(len(colors)))
                ax.set_xticklabels([f"Color {i + 1}" for i in range(len(colors))])
                ax.set_ylabel("Proportion (%)")
                ax.set_title("Color Proportions in Saree")
                st.pyplot(fig)

                # Display HEX codes, RGB values, and proportions
                st.write("### Dominant Colors:")
                for i, (color, hex_code, proportion) in enumerate(zip(colors, hex_codes, proportions)):
                    st.markdown(f"**Color {i + 1}:** HEX: `{hex_code}` | RGB: {color} | Proportion: {proportion:.2f}%")

            except Exception as e:
                st.error(f"An error occurred: {e}")

    else:
        st.info("Please upload an image to begin.")

if __name__ == "__main__":
    main()
