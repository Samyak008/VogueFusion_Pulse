import streamlit as st
import numpy as np
import json

# Load dye details from JSON file
def load_dye_details():
    with open('data/dye_detail.json') as f:
        dye_details = json.load(f)
    return dye_details

# Function to calculate the cost for a given fabric gsm
def calculate_cost(fabric_gsm, total_color_weight=7):
    # Adjust total weight based on fabric gsm
    total_weight = total_color_weight if fabric_gsm == 70 else (fabric_gsm / 70) * total_color_weight
    cost_per_color = []

    # Retrieve the best proportions, selected dyes, and counts from session state
    selected_dyes = st.session_state.selected_dyes
    best_proportions = st.session_state.best_proportions
    dyes = load_dye_details()  # Load the dye details from the JSON file
    colors = st.session_state.colors
    counts = st.session_state.counts

    total_pixels = sum(counts)

    # Calculate the cost for each color based on its count proportion
    for i, target_rgb in enumerate(colors):
        color_cost = 0
        # Find the proportion for the current color
        color_proportion = counts[i] / total_pixels  # Proportion of the color in total design
        color_weight = total_weight * color_proportion  # Weight of the color in the design
        
        # For each dye, calculate the cost based on its proportion and color weight
        for j, dye in enumerate(dyes):
            if dye["color"] in selected_dyes:
                dye_weight = color_weight * best_proportions[j]  # Weight of this dye in the color
                dye_cost = dye_weight * dye["price"]  # Cost for this dye based on weight and price
                color_cost += dye_cost
        
        cost_per_color.append(color_cost)
    
    return cost_per_color

# Streamlit app for cost prediction
def main():
    st.title("Saree Dye Cost Prediction")

    # Select fabric GSM for cost calculation
    fabric_gsm = st.selectbox("Select Fabric GSM", [70, 100, 120])

    # Retrieve the results from the genetic algorithm
    if "selected_dyes" in st.session_state and "best_proportions" in st.session_state:
        # Display the dye optimization results
        total_cost = 0
        cost_per_color = calculate_cost(fabric_gsm)

        for i, target_rgb in enumerate(st.session_state.colors):
            st.write(f"#### Color {i + 1}")
            best_proportions = st.session_state.best_proportions
            selected_dyes = st.session_state.selected_dyes
            resulting_rgb = np.dot(best_proportions, [dye["rgb"] for dye in selected_dyes]).round().astype(int)

            st.write(f"**Target RGB:** {target_rgb}")
            st.write(f"**Resulting RGB:** {resulting_rgb}")
            st.write("**Optimal Dye Proportions:**")

            color_cost = cost_per_color[i]
            for dye, proportion in zip(st.session_state.dyes, best_proportions):
                if proportion > 0.01:  # Filter negligible proportions
                    st.markdown(f"**{dye['color']}:**")
                    st.write(f"Proportion: {proportion:.2%}")
                    st.progress(proportion)  # Show progress bar for proportion
                    st.markdown(f"<div style='background-color: rgb{tuple(dye['rgb'])}; width: 100px; height: 30px;'></div>", unsafe_allow_html=True)
                    st.write(f"Cost for {dye['color']}: ₹{(color_cost * proportion):.2f}")
                    st.write("")

            # Display the cost for the color
            st.write(f"**Total Cost for Color {i + 1}: ₹{color_cost:.2f}")
            total_cost += color_cost
            st.write("---")
        
        # Display the total cost of the design
        st.write(f"**Total Cost for the Entire Design: ₹{total_cost:.2f}")

    else:
        st.error("Please ensure that the dye proportions and selected dyes are available.")

if __name__ == "__main__":
    main()
