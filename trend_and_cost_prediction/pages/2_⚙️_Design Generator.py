import streamlit as st

def main():
    # Set the title for the Streamlit app
    st.title("Design Generation")

    # Description for the app
    st.write("Welcome to the Design Generator! Click the button below to be redirected to the design generation model.")

    # URL of the design generation model
    model_url = "https://example.com/design-generator"  # Replace with your actual model URL

    # Button to redirect to the URL
    if st.button("Go to Design Generator"):
        st.write(f"[Click here to open the Design Generator]({model_url})")
        st.markdown(f"""<a href='{model_url}' target='_blank' style='text-decoration:none;'>
                    <button style='background-color:#4CAF50;color:white;padding:10px 20px;border:none;border-radius:4px;'>
                    Open Design Generator
                    </button></a>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
