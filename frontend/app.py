import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import requests
import json

st.title("AI Med - Digit Recognizer PoC")

st.markdown("Draw a digit (0-9) inside the box below and click **Predict**.")

# Create a canvas component
# MNIST images are white on black, so we use white stroke on black background
canvas_result = st_canvas(
    fill_color="black",  # Fixed fill color with some opacity
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Get image data from canvas
        # canvas_result.image_data is RGBA
        input_image = canvas_result.image_data.astype('uint8')
        
        # Convert to grayscale
        # We can just take the first channel if it's black and white, or use cv2
        if input_image.shape[2] == 4:
            input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_RGBA2GRAY)
        else:
            input_image_gray = input_image
            
        # Resize to 28x28
        image_resized = cv2.resize(input_image_gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Prepare payload
        payload = {"image": image_resized.tolist()}
        
        # Send request to backend
        backend_url = "http://backend:8000/predict"
        
        try:
            with st.spinner("Predicting..."):
                response = requests.post(backend_url, json=payload)
                
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                confidence = result["confidence"]
                
                st.success(f"Prediction: **{prediction}**")
                st.info(f"Confidence: {confidence:.4f}")
                
                # Debug: Show what the model sees
                st.write("Model Input (28x28):")
                st.image(image_resized, width=100, clamp=True)
                
            else:
                st.error(f"Error: {response.status_code}")
                st.write(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend. Is it running?")
            st.write(f"Attempted to connect to: `{backend_url}`")
    else:
        st.warning("Please draw something first.")
