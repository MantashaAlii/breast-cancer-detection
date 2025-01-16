import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Define image dimensions
image_width = 224
image_height = 224

# Load the trained model
model = load_model("breast_cancer_detection_model.h5")

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), -1)  # Convert to numpy array
    image = cv2.resize(image, (image_width, image_height))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR format
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.0
    return image

# Function to make predictions
def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return prediction

# Main Streamlit app
def main():
    st.title("Breast Cancer Detection")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        # Make prediction when the user clicks the button
        if st.button('Predict'):
            # Make prediction
            prediction = predict(uploaded_image)

            # Display prediction result
            if prediction[0][0] > prediction[0][1]:
                st.success("The image is classified as Benign.")
            else:
                st.success("The image is classified as Malignant.")

if __name__ == "__main__":
    main()

