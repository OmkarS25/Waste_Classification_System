# ============================Basic=================================

# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import cv2
# import time
# import os
# import pandas as pd
# from datetime import datetime

# # Load your trained model
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model('./final_model.h5')

# def preprocess_image(image):
#     """Preprocess image for model prediction."""
#     img = image.resize((299, 299))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img, img_array

# def predict(image):
#     """Predict the class of the image."""
#     model = load_model()
#     _, processed_image = preprocess_image(image)
#     prediction = model.predict(processed_image)
#     class_names = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']
#     return {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}

# def log_prediction(image_path, prediction):
#     """Log predictions with timestamp."""
#     log_data = {
#         "timestamp": [datetime.now()],
#         "image_path": [image_path],
#         "predictions": [prediction]
#     }
#     log_df = pd.DataFrame(log_data)
#     if os.path.exists("prediction_log.csv"):
#         log_df.to_csv("prediction_log.csv", mode="a", index=False, header=False)
#     else:
#         log_df.to_csv("prediction_log.csv", index=False)

# def display_prediction_results(prediction):
#     """Display prediction results with progress bars and bar charts."""
#     predicted_class = max(prediction, key=prediction.get)
#     confidence = prediction[predicted_class]
    
#     st.write(f"Predicted waste type: **{predicted_class}**")
#     st.write(f"Confidence: {confidence:.2%}")

#     # Visualize confidence levels with progress bars
#     st.subheader("Confidence Levels:")
#     for waste_type, prob in prediction.items():
#         st.progress(int(prob * 100), text=f"{waste_type}: {prob:.2%}")

#     # Bar chart visualization
#     fig = go.Figure(data=[go.Bar(
#         x=list(prediction.keys()),
#         y=list(prediction.values()),
#         marker=dict(color=list(prediction.values()), colorscale='Viridis', colorbar=dict(title='Probability'))
#     )])
#     fig.update_layout(
#         title='Prediction Probabilities',
#         xaxis_title='Waste Type',
#         yaxis_title='Probability',
#         height=500,
#         width=700
#     )
#     st.plotly_chart(fig)

# def run_camera_realtime():
#     """Run real-time waste classification using the camera."""
#     st.title('📷 Live Waste Classification (Real-Time)')
#     run_camera = st.checkbox('Start Camera Feed')
#     model = load_model()

#     if run_camera:
#         cap = cv2.VideoCapture(0)
#         stframe = st.empty()

#         while run_camera:
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("Failed to access the camera.")
#                 break

#             # Convert frame to PIL image for prediction
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(frame_rgb)
#             _, processed_image = preprocess_image(image)
#             prediction = model.predict(processed_image)
#             class_names = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']
#             predicted_class = class_names[np.argmax(prediction)]
#             confidence = np.max(prediction)

#             # Overlay prediction on the frame
#             cv2.putText(frame_rgb, f"{predicted_class} ({confidence:.2%})", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#             # Display the frame with predictions
#             stframe.image(frame_rgb, channels='RGB', use_column_width=True)

#         cap.release()
#         cv2.destroyAllWindows()

# def run():
#     """Main application."""
#     st.title("🔍 Advanced Waste Classification App")
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Choose a feature:", ["Home", "Upload Image", "Camera (Real-Time)", "Logs"])

#     if page == "Home":
#         st.header("Welcome!")
#         st.write("""
#         This app uses a machine learning model to classify waste types. 
#         You can upload an image, use example images, or leverage real-time camera feed.
#         """)
    
#     elif page == "Upload Image":
#         st.header("Upload an Image")
#         uploaded_file = st.file_uploader("Upload an image of waste (JPG/PNG)", type=["jpg", "jpeg", "png"])
        
#         if uploaded_file:
#             image = Image.open(uploaded_file).convert("RGB")
#             st.image(image, caption="Uploaded Image", use_column_width=True)
#             if st.button("Predict"):
#                 prediction = predict(image)
#                 display_prediction_results(prediction)
#                 log_prediction(uploaded_file.name, prediction)
    
#     elif page == "Camera (Real-Time)":
#         run_camera_realtime()
    
#     elif page == "Logs":
#         st.header("Prediction Logs")
#         if os.path.exists("prediction_log.csv"):
#             logs = pd.read_csv("prediction_log.csv")
#             st.dataframe(logs)
#         else:
#             st.info("No logs available yet.")

# if __name__ == "__main__":
#     run()








# ============================Intermediate=================================

import streamlit as st
import numpy as np
from PIL import Image
# import plotly.graph_objects as go
import tensorflow as tf
import cv2
import time
import os
import pandas as pd
from datetime import datetime
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import folium
from streamlit_folium import folium_static
import plotly.graph_objects as go


# Load your trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('./final_model.h5')

def preprocess_image(image):
    """Preprocess image for model prediction."""
    img = image.resize((299, 299))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def predict(image):
    """Predict the class of the image."""
    model = load_model()
    _, processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_names = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']
    return {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}

def log_prediction(image_path, prediction, user_feedback=None):
    """Log predictions with timestamp and user feedback."""
    log_data = {
        "timestamp": [datetime.now()],
        "image_path": [image_path],
        "predictions": [prediction],
        "user_feedback": [user_feedback]
    }
    log_df = pd.DataFrame(log_data)
    if os.path.exists("prediction_log.csv"):
        log_df.to_csv("prediction_log.csv", mode="a", index=False, header=False)
    else:
        log_df.to_csv("prediction_log.csv", index=False)

def display_prediction_results(prediction):
    """Display prediction results with progress bars and bar charts."""
    predicted_class = max(prediction, key=prediction.get)
    confidence = prediction[predicted_class]
    
    st.write(f"Predicted waste type: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2%}")

    # Visualize confidence levels with progress bars
    st.subheader("Confidence Levels:")
    for waste_type, prob in prediction.items():
        st.progress(int(prob * 100), text=f"{waste_type}: {prob:.2%}")

    # Bar chart visualization
    fig = go.Figure(data=[go.Bar(
        x=list(prediction.keys()),
        y=list(prediction.values()),
        marker=dict(color=list(prediction.values()), colorscale='Viridis', colorbar=dict(title='Probability'))
    )])
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Waste Type',
        yaxis_title='Probability',
        height=500,
        width=700
    )
    st.plotly_chart(fig)

    # User feedback
    feedback = st.radio("Was the prediction correct?", ("Correct", "Incorrect"), index=None)
    if feedback:
        log_prediction("uploaded_image", prediction, feedback)
        st.success("Thank you for your feedback!")

    # Gamification: Earn points
    if feedback == "Correct":
        st.session_state.points += 10
        st.write(f"🎉 You earned 10 points! Total points: {st.session_state.points}")
    elif feedback == "Incorrect":
        st.write("😢 No points this time. Keep trying!")

    # Educational content
    st.subheader("Did You Know?")
    if predicted_class == "Plastic":
        st.write("Plastic can take up to 500 years to decompose. Always recycle!")
    elif predicted_class == "Paper":
        st.write("Recycling one ton of paper saves 17 trees. Great job!")

def explain_prediction(image, model):
    """Explain model prediction using SHAP."""
    st.subheader("Model Explainability with SHAP")
    _, processed_image = preprocess_image(image)
    explainer = shap.DeepExplainer(model, np.zeros((1, 299, 299, 3)))
    shap_values = explainer.shap_values(processed_image)
    shap.image_plot(shap_values, -processed_image)

def run_camera_realtime():
    """Run real-time waste classification using the camera."""
    st.title('📷 Live Waste Classification (Real-Time)')
    run_camera = st.checkbox('Start Camera Feed')
    model = load_model()

    if run_camera:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access the camera.")
                break

            # Convert frame to PIL image for prediction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            _, processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            class_names = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Overlay prediction on the frame
            cv2.putText(frame_rgb, f"{predicted_class} ({confidence:.2%})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the frame with predictions
            stframe.image(frame_rgb, channels='RGB', use_container_width=True)

        cap.release()
        cv2.destroyAllWindows()

def waste_disposal_locator():
    """Show nearby waste disposal facilities on a map."""
    st.title("🗺️ Waste Disposal Locator")
    st.write("Find nearby recycling centers, waste disposal facilities, or composting sites.")

    # Example map (replace with real data)
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=12)  # San Francisco coordinates
    folium.Marker([37.7749, -122.4194], popup="Recycling Center").add_to(m)
    folium.Marker([37.7849, -122.4294], popup="Composting Site").add_to(m)
    folium_static(m)

def waste_reduction_tips():
    """Provide personalized waste reduction tips."""
    st.title("💡 Waste Reduction Tips")
    st.write("Here are some tips to help you reduce waste:")

    tips = {
        "Plastic": "Use reusable bags and bottles instead of single-use plastics.",
        "Paper": "Go digital with bills and documents to reduce paper usage.",
        "Food Organics": "Compost food scraps to reduce landfill waste.",
        "Glass": "Recycle glass bottles and jars instead of throwing them away.",
        "Metal": "Recycle aluminum cans and metal containers.",
    }

    for category, tip in tips.items():
        st.write(f"**{category}**: {tip}")

def community_challenges():
    """Show community challenges and progress."""
    st.title("🏆 Community Challenges")
    st.write("Join community-driven challenges to make a collective impact!")

    challenges = {
        "Recycle 100 Plastic Items": "30% completed",
        "Compost 50 Food Scraps": "15% completed",
        "Reduce Paper Usage by 20%": "10% completed",
    }

    for challenge, progress in challenges.items():
        st.write(f"**{challenge}**: {progress}")

def carbon_footprint_calculator():
    """Calculate the user's carbon footprint."""
    st.title("🌍 Carbon Footprint Calculator")
    st.write("Calculate your carbon footprint based on your waste disposal habits.")

    # Example calculation (replace with real logic)
    plastic_usage = st.slider("How much plastic do you use per week (in kg)?", 0, 10, 1)
    paper_usage = st.slider("How much paper do you use per week (in kg)?", 0, 10, 1)
    food_waste = st.slider("How much food waste do you generate per week (in kg)?", 0, 10, 1)

    carbon_footprint = (plastic_usage * 3.8) + (paper_usage * 1.5) + (food_waste * 2.1)
    st.write(f"Your estimated carbon footprint is **{carbon_footprint:.2f} kg CO2 per week**.")

def update_sidebar():
    """Update sidebar with navigation options."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a feature:", ["📹 Live Detection", "Upload Image", "Camera (Real-Time)", "Logs", "Leaderboard", "Waste Disposal Locator", "Waste Reduction Tips", "Community Challenges", "Carbon Footprint Calculator"])
    return page

def run():
    """Main application."""
    st.title("🔍 Advanced Waste Classification App")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a feature:", ["📹 Live Detection", "Upload Image", "Camera (Real-Time)", "Logs", "Leaderboard", "Waste Disposal Locator", "Waste Reduction Tips", "Community Challenges", "Carbon Footprint Calculator"])

    # Initialize session state for gamification
    if "points" not in st.session_state:
        st.session_state.points = 0

    if page == "📹 Live Detection":
        st.header("Welcome!")
        st.write("""
        This app uses a machine learning model to classify waste types. 
        You can upload an image, use example images, or leverage real-time camera feed.
        """)
    
    elif page == "Upload Image":
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Upload an image of waste (JPG/PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Predict"):
                prediction = predict(image)
                display_prediction_results(prediction)
                explain_prediction(image, load_model())
    
    elif page == "Camera (Real-Time)":
        run_camera_realtime()
    
    elif page == "Logs":
        st.header("Prediction Logs")
        if os.path.exists("prediction_log.csv"):
            logs = pd.read_csv("prediction_log.csv")
            st.dataframe(logs)
            st.download_button("Download Logs", logs.to_csv(index=False).encode('utf-8'), file_name="prediction_log.csv")
        else:
            st.info("No logs available yet.")
    
    elif page == "Leaderboard":
        st.header("Leaderboard")
        st.write("🏆 Top Recyclers:")
        # Example leaderboard (replace with real data)
        leaderboard_data = {
            "User": ["Alice", "Bob", "Charlie"],
            "Points": [150, 120, 90]
        }
        leaderboard_df = pd.DataFrame(leaderboard_data)
        st.dataframe(leaderboard_df)
    
    elif page == "Waste Disposal Locator":
        waste_disposal_locator()
    
    elif page == "Waste Reduction Tips":
        waste_reduction_tips()
    
    elif page == "Community Challenges":
        community_challenges()
    
    elif page == "Carbon Footprint Calculator":
        carbon_footprint_calculator()

if __name__ == "__main__":
    run()










# ============================Advance=================================
# import streamlit as st
# import numpy as np
# from PIL import Image
# import plotly.graph_objects as go
# import tensorflow as tf
# import cv2
# import os
# import pandas as pd
# from datetime import datetime
# import shap
# import folium
# from streamlit_folium import folium_static
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter
# import requests  # Import for API requests

# # Load your trained model
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model('./final_model.h5')

# def preprocess_image(image):
#     """Preprocess image for model prediction."""
#     img = image.resize((299, 299))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img, img_array

# def predict(image):
#     """Predict the class of the image."""
#     model = load_model()
#     _, processed_image = preprocess_image(image)
#     prediction = model.predict(processed_image)
#     class_names = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']
#     return {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}

# def waste_disposal_locator():
#     """Show nearby waste disposal facilities on a map using real-time API data."""
#     st.title("🗺️ Waste Disposal Locator")
#     st.write("Find nearby recycling centers, waste disposal facilities, or composting sites in real-time.")

#     # User input for location
#     location_input = st.text_input("Enter a location (e.g., city, address, or coordinates):", "San Francisco, CA")
#     search_radius = st.slider("Search radius (in kilometers):", 1, 50, 10)

#     if st.button("Search"):
#         with st.spinner("Fetching real-time data..."):
#             try:
#                 # Initialize geolocator
#                 geolocator = Nominatim(user_agent="waste_locator_app")
#                 geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

#                 # Get coordinates for the user's input location
#                 location = geocode(location_input)
#                 if location:
#                     st.success(f"Location found: {location.address}")
#                     lat, lon = location.latitude, location.longitude

#                     # Create a map centered at the user's location
#                     m = folium.Map(location=[lat, lon], zoom_start=12)

#                     # Use Google Places API to search for nearby recycling centers
#                     # Replace 'YOUR_GOOGLE_API_KEY' with your actual Google API Key
#                     GOOGLE_API_KEY = "980060f74703487581edb644c3284a87"
#                     places_url = (
#                         f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
#                         f"?location={lat},{lon}&radius={search_radius * 1000}"
#                         f"&type=recycling&key={GOOGLE_API_KEY}"
#                     )
#                     response = requests.get(places_url)
#                     places_data = response.json()

#                     if "results" in places_data and len(places_data["results"]) > 0:
#                         st.info(f"Found {len(places_data['results'])} recycling facilities nearby.")
#                         for place in places_data["results"]:
#                             name = place["name"]
#                             place_lat = place["geometry"]["location"]["lat"]
#                             place_lon = place["geometry"]["location"]["lng"]
#                             address = place.get("vicinity", "No address available")

#                             # Add marker for each place on the map
#                             folium.Marker(
#                                 location=[place_lat, place_lon],
#                                 popup=f"{name}\n{address}",
#                                 icon=folium.Icon(color="green", icon="recycle", prefix="fa")
#                             ).add_to(m)
#                     else:
#                         st.warning("No recycling centers found nearby. Try expanding the search radius.")

#                     # Display the map
#                     folium_static(m)
#                 else:
#                     st.error("Location not found. Please try again.")
#             except Exception as e:
#                 st.error(f"An error occurred: {e}")

# def run():
#     """Main application."""
#     st.title("🔍 Advanced Waste Classification App")
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Choose a feature:", ["Home", "Upload Image", "Waste Disposal Locator"])

#     if page == "Home":
#         st.header("Welcome!")
#         st.write("""This app uses a machine learning model to classify waste types 
#         and provides tools like a waste disposal locator.""")
    
#     elif page == "Waste Disposal Locator":
#         waste_disposal_locator()

# if __name__ == "__main__":
#     run()
















# ----------------------------------------Perfect Executable---------------------------------------------------
# import streamlit as st
# import numpy as np
# from PIL import Image
# import plotly.graph_objects as go
# import tensorflow as tf
# import cv2
# import os
# import pandas as pd
# from datetime import datetime
# import shap
# import folium
# from streamlit_folium import folium_static
# import requests

# # Load your trained model
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model('./final_model.h5')

# def preprocess_image(image):
#     """Preprocess image for model prediction."""
#     img = image.resize((299, 299))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img, img_array

# def predict(image):
#     """Predict the class of the image."""
#     model = load_model()
#     _, processed_image = preprocess_image(image)
#     prediction = model.predict(processed_image)
#     class_names = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']
#     return {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}

# def log_prediction(image_path, prediction, user_feedback=None):
#     """Log predictions with timestamp and user feedback."""
#     log_data = {
#         "timestamp": [datetime.now()],
#         "image_path": [image_path],
#         "predictions": [prediction],
#         "user_feedback": [user_feedback]
#     }
#     log_df = pd.DataFrame(log_data)
#     if os.path.exists("prediction_log.csv"):
#         log_df.to_csv("prediction_log.csv", mode="a", index=False, header=False)
#     else:
#         log_df.to_csv("prediction_log.csv", index=False)

# def display_prediction_results(prediction):
#     """Display prediction results with progress bars and bar charts."""
#     predicted_class = max(prediction, key=prediction.get)
#     confidence = prediction[predicted_class]
    
#     st.success(f"Predicted waste type: **{predicted_class}**")
#     st.info(f"Confidence: {confidence:.2%}")

#     # Visualize confidence levels with progress bars
#     st.subheader("Confidence Levels:")
#     for waste_type, prob in prediction.items():
#         st.progress(int(prob * 100), text=f"{waste_type}: {prob:.2%}")

#     # Bar chart visualization
#     fig = go.Figure(data=[go.Bar(
#         x=list(prediction.keys()),
#         y=list(prediction.values()),
#         marker=dict(color=list(prediction.values()), colorscale='Viridis', colorbar=dict(title='Probability'))
#     )])
#     fig.update_layout(
#         title='Prediction Probabilities',
#         xaxis_title='Waste Type',
#         yaxis_title='Probability',
#         height=500,
#         width=700
#     )
#     st.plotly_chart(fig)

#     # User feedback
#     feedback = st.radio("Was the prediction correct?", ("Correct", "Incorrect"), index=None)
#     if feedback:
#         log_prediction("uploaded_image", prediction, feedback)
#         st.success("Thank you for your feedback!")

#     # Gamification: Earn points
#     if feedback == "Correct":
#         st.session_state.points += 10
#         st.balloons()
#         st.success(f"🎉 You earned 10 points! Total points: {st.session_state.points}")
#     elif feedback == "Incorrect":
#         st.warning("😢 No points this time. Keep trying!")

#     # Educational content
#     st.subheader("Did You Know?")
#     if predicted_class == "Plastic":
#         st.info("Plastic can take up to 500 years to decompose. Always recycle!")
#     elif predicted_class == "Paper":
#         st.info("Recycling one ton of paper saves 17 trees. Great job!")

# def explain_prediction(image, model):
#     """Explain model prediction using SHAP."""
#     st.subheader("Model Explainability with SHAP")
#     _, processed_image = preprocess_image(image)
#     explainer = shap.DeepExplainer(model, np.zeros((1, 299, 299, 3)))
#     shap_values = explainer.shap_values(processed_image)
#     shap.image_plot(shap_values, -processed_image)

# def run_camera_realtime():
#     """Run real-time waste classification using the camera."""
#     st.title('📷 Live Waste Classification (Real-Time)')
#     run_camera = st.checkbox('Start Camera Feed')
#     model = load_model()

#     if run_camera:
#         cap = cv2.VideoCapture(0)
#         stframe = st.empty()

#         while run_camera:
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("Failed to access the camera.")
#                 break

#             # Convert frame to PIL image for prediction
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(frame_rgb)
#             _, processed_image = preprocess_image(image)
#             prediction = model.predict(processed_image)
#             class_names = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']
#             predicted_class = class_names[np.argmax(prediction)]
#             confidence = np.max(prediction)

#             # Overlay prediction on the frame
#             cv2.putText(frame_rgb, f"{predicted_class} ({confidence:.2%})", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#             # Display the frame with predictions
#             stframe.image(frame_rgb, channels='RGB', use_column_width=True)

#         cap.release()
#         cv2.destroyAllWindows()

# def get_lat_lng(address):
#     """Get latitude and longitude for a given address using OpenCage Geocoder API."""
#     api_key = "980060f74703487581edb644c3284a87"  # Replace with your OpenCage API key
#     url = f"https://api.opencagedata.com/geocode/v1/json?q={address}&key={api_key}"

#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         data = response.json()

#         if data['results']:
#             location = data['results'][0]['geometry']
#             return location['lat'], location['lng']
#         else:
#             st.error("No results found for the given address.")
#             return None, None
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error fetching data from OpenCage API: {e}")
#         return None, None

# def waste_disposal_locator():
#     """Show nearby waste disposal facilities on a map using OpenCage API."""
#     st.title("🗺️ Waste Disposal Locator")
#     st.write("Find nearby recycling centers, waste disposal facilities, or composting sites in real-time.")

#     # User input for location
#     location_input = st.text_input("Enter a location (e.g., city, address, or coordinates):", "San Francisco, CA")

#     if st.button("Search"):
#         with st.spinner("Fetching data..."):
#             lat, lon = get_lat_lng(location_input)
#             if lat and lon:
#                 st.success(f"Location found: {location_input}")

#                 # Create a map centered at the user's location
#                 m = folium.Map(location=[lat, lon], zoom_start=12)

#                 # Add marker for user location
#                 folium.Marker(
#                     location=[lat, lon],
#                     popup=f"{location_input}",
#                     icon=folium.Icon(color="blue")
#                 ).add_to(m)

#                 # Display the map
#                 folium_static(m)
#             else:
#                 st.error("Unable to find the location. Please try again.")

# def run():
#     """Main application."""
#     st.title("🔍 Advanced Waste Classification App")
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Choose a feature:", ["Home", "Upload Image", "Camera (Real-Time)", "Logs", "Leaderboard", "Waste Disposal Locator", "Waste Reduction Tips", "Community Challenges", "Carbon Footprint Calculator"])

#     # Initialize session state for gamification
#     if "points" not in st.session_state:
#         st.session_state.points = 0

#     if page == "Home":
#         st.header("Welcome!")
#         st.write("Welcome to the Advanced Waste Classification App. Choose an option from the sidebar to get started.")

#     elif page == "Upload Image":
#         st.header("Upload Image for Waste Classification")
#         uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
#         if uploaded_image is not None:
#             image = Image.open(uploaded_image)
#             prediction = predict(image)
#             display_prediction_results(prediction)

#     elif page == "Camera (Real-Time)":
#         run_camera_realtime()

#     elif page == "Waste Disposal Locator":
#         waste_disposal_locator()

#     # Add placeholders for the other features like Leaderboard, Waste Reduction Tips, etc.
#     elif page == "Leaderboard":
#         st.header("Leaderboard")
#         st.write("Leaderboard feature coming soon!")

#     elif page == "Waste Reduction Tips":
#         st.header("Waste Reduction Tips")
#         st.write("Learn how to reduce your waste! Coming soon.")

#     elif page == "Community Challenges":
#         st.header("Community Challenges")
#         st.write("Join challenges to reduce waste in your community. Coming soon.")

#     elif page == "Carbon Footprint Calculator":
#         st.header("Carbon Footprint Calculator")
#         st.write("Calculate your carbon footprint. Coming soon.")

# if __name__ == "__main__":
#     run()