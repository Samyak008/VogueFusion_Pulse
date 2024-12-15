# Step 1: Install dependencies from requirements.txt
Write-Host "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 2: Navigate to the directory where the Streamlit app is located
# Replace "C:\path\to\your\streamlit\app" with the actual path to your Streamlit app directory
Set-Location -Path ".\trend_and_cost_prediction"

# Step 3: Run the Streamlit app
Write-Host "Running Streamlit app..."
streamlit run VogueFusion.py
