import streamlit as st
import json
import numpy as np

# Load dye details from JSON file
def load_dye_details():
    with open('data/dye_detail.json') as f:
        dye_details = json.load(f)
    return dye_details

# Function to calculate the cost for a given fabric gsm
def calculate_cost(fabric_gsm, total_color_weight=0.7):
    # Adjust total weight based on fabric gsm
    total_weight = total_color_weight if fabric_gsm == 70 else (fabric_gsm / 70) * total_color_weight
    cost_per_color = []

    # Retrieve the best proportions and dye details from session state and load the dye details
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

        # For each dye and its proportion, calculate the cost based on the dye weight
        for dye, proportion in zip(dyes, best_proportions):
            if proportion > 0.01:  # Filter negligible proportions
                dye_weight = color_weight * proportion  # Weight of this dye in the color
                dye_cost = dye_weight * dye["price"]  # Cost for this dye based on weight and price
                color_cost += dye_cost  # Accumulate the cost for the color

        cost_per_color.append(color_cost)  # Add the color cost to the list
    
    return cost_per_color


# Streamlit app for cost prediction
def main():
    st.title("Saree Dye Cost Prediction")
    
    # Display instructions for the user
    st.info("This app predicts the total cost of dyeing a saree based on selected fabric GSM and dye proportions.")
    
    # Select fabric GSM for cost calculation
    fabric_gsm = st.selectbox("Select Fabric GSM", [70, 100, 120], index=0)

    # Check if the best proportions are available in the session state
    if "best_proportions" in st.session_state:
        # Display the dye optimization results
        st.subheader("Dye Proportions and Cost Calculation")
        st.write("Calculating cost based on selected fabric GSM and dye proportions...")

        # Calculate the cost per color based on fabric GSM
        cost_per_color = calculate_cost(fabric_gsm)

        total_cost = 0

        # Display each color's cost with a more visual presentation
        for i, color_cost in enumerate(cost_per_color):
            st.markdown(f"### Total Cost for Color {i + 1}: ₹{color_cost:.2f}")
            total_cost += color_cost  # Add the current color's cost to the total cost
            st.write("---")

        # Display the total cost for the entire design
# Display the total cost for the entire design in a single line
        st.markdown(f"### Total Cost for the Entire Design: **₹{total_cost:.2f}**")


        # Display the dye proportions with color blocks and progress bars
        st.subheader("Dye Proportions:")
        dyes = load_dye_details()
        for dye, proportion in zip(dyes, st.session_state.best_proportions):
            if proportion > 0.01:  # Filter negligible proportions
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{dye['color']}:** {proportion:.2%} of the design")
                with col2:
                    st.progress(proportion)  # Show progress bar for proportion
                
                # Show color block with corresponding dye color
                st.markdown(f"<div style='background-color: rgb{tuple(dye['rgb'])}; width: 100px; height: 30px;'></div>", unsafe_allow_html=True)
                st.write("")

    else:
        st.error("Please ensure that the dye proportions and selected dyes are available.")

if __name__ == "__main__":
    main()
