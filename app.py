# app.py

import streamlit as st
import home
import eda
import prediction
import live_detection
import live_detection_extra

# Set the page configuration for the Streamlit app
st.set_page_config(
    page_title="Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Create a sidebar for navigation
    st.sidebar.title("♻️ Eco Lens")
    page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA (How data was analized?)", "🔍 Prediction", "📹 Live Detection"])

    # Display the selected page content
    if page == "🏠 Home":
        # Run the Home module
        home.run()


    elif page == "📊 EDA (How data was analized?)":
        # Run the Exploratory Data Analysis module
        eda.run()
    
    elif page == "🔍 Prediction":
        # Run the Prediction module
        prediction.run()
    
    elif page == "📹 Live Detection":
        # Run the Live Detection module
        live_detection.run_camera_realtime()


if __name__ == "__main__":
    main()