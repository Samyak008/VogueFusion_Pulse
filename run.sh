#!/bin/bash

# Step 1: Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 2: Navigate to the directory where the Streamlit app is located
# Replace "/path/to/your/streamlit/app" with the actual directory of your Streamlit app
cd ./trend_and_cost_prediction || exit

# Step 3: Run the Streamlit app
echo "Running Streamlit app..."
streamlit run VogueFusion.py
