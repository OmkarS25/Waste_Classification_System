import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('./final_model.h5')

# Preprocess image for the model
def preprocess_image(image):
    img = cv2.resize(image, (299, 299))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Perform prediction
def predict(image, model):
    class_names = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 
                   'Paper', 'Plastic', 'Textile Trash', 'Vegetation']
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return predicted_class, confidence

# Live detection function
def live_detection():
    st.title("📹 Live Waste Detection")
    st.write("Turn on your camera, and the system will classify waste objects in real time.")

    # Load the model
    model = load_model()

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)  # 0 for default camera

    if not video_capture.isOpened():
        st.error("Unable to access the camera.")
        return

    # Streamlit's interactive video feed
    stframe = st.empty()  # Create an empty frame for video display

    # Start live detection
    while True:
        ret, frame = video_capture.read()

        if not ret:
            st.warning("Failed to capture video frame.")
            break

        # Resize the frame for prediction (optional)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        h, w, _ = frame_rgb.shape

        # Object detection logic: (Can be replaced with YOLO/SSD)
        step = 200  # Region size for cropping
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Crop region
                region = frame_rgb[y:y+step, x:x+step]
                if region.shape[0] == 0 or region.shape[1] == 0:
                    continue

                # Predict the class for the cropped region
                predicted_class, confidence = predict(region, model)

                # Draw a bounding box and label
                cv2.rectangle(frame, (x, y), (x+step, y+step), (0, 255, 0), 2)
                label = f"{predicted_class} ({confidence:.2%})"
                cv2.putText(frame, label, (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the processed frame
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Break the loop if the "Stop" button is pressed
        if st.button("Stop Camera"):
            break

    # Release the camera and clean up
    video_capture.release()
    cv2.destroyAllWindows()
    st.write("Live detection stopped.")

# Add the Live Detection page to Streamlit
def run():
    live_detection()

if __name__ == "__main__":
    run()


# Version 2: Add Object Detection with YOLOv5

# import cv2
# import torch
# import numpy as np
# import streamlit as st
# from PIL import Image
# from tensorflow.keras.models import load_model

# # Load YOLOv5 model for object detection
# @st.cache_resource
# def load_detection_model():
#     return torch.hub.load('ultralytics/yolov5', 'yolov5l')  # YOLOv5 Large model

# # Load waste classification model
# @st.cache_resource
# def load_classification_model():
#     return load_model('./final_model.h5')  # Pre-trained lightweight classification model

# # Preprocess the detected objects for classification
# def preprocess_for_classification(cropped_image):
#     img = cv2.resize(cropped_image, (128, 128))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # Perform object detection
# def detect_objects(frame, detection_model):
#     results = detection_model(frame)
#     return results.pandas().xyxy[0]  # Bounding box dataframe

# # Perform waste classification
# def classify_objects(cropped_image, classification_model):
#     class_names = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 
#                    'Paper', 'Plastic', 'Textile Trash', 'Vegetation']
#     processed_image = preprocess_for_classification(cropped_image)
#     prediction = classification_model.predict(processed_image)
#     predicted_class = class_names[np.argmax(prediction[0])]
#     confidence = np.max(prediction[0])
#     return predicted_class, confidence

# # Streamlit app for live detection
# def live_detection():
#     st.title("🔍 Real-Time Waste Detection")
#     st.write("This system detects objects and classifies waste categories in real-time.")

#     # Load models
#     detection_model = load_detection_model()
#     classification_model = load_classification_model()

#     # Initialize webcam
#     video_capture = cv2.VideoCapture(0)
#     if not video_capture.isOpened():
#         st.error("Unable to access the camera.")
#         return

#     # Streamlit frame for video feed
#     stframe = st.empty()

#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             st.warning("Failed to capture video frame.")
#             break

#         # Perform object detection
#         detection_results = detect_objects(frame, detection_model)

#         # Process each detected object
#         for _, row in detection_results.iterrows():
#             # Get bounding box coordinates
#             x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
#             cropped_img = frame[y_min:y_max, x_min:x_max]

#             # Classify the cropped object
#             try:
#                 predicted_class, confidence = classify_objects(cropped_img, classification_model)
#             except Exception as e:
#                 continue

#             # Draw bounding box and label
#             label = f"{predicted_class} ({confidence:.2%})"
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # Display the processed frame
#         stframe.image(frame, channels="BGR", use_column_width=True)

#         # Stop camera if "Stop" button is pressed
#         if st.button("Stop Camera"):
#             break

#     # Release resources
#     video_capture.release()
#     cv2.destroyAllWindows()

# # Add the Live Detection page
# def run():
#     live_detection()

# if __name__ == "__main__":
#     run()
