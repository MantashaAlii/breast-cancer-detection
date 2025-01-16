import cv2
from keras.models import load_model
import numpy as np

# Define image dimensions
image_width = 224
image_height = 224

# Load the pre-trained model
model = load_model("breast_cancer_detection_model.h5")

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_width, image_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

# Test image path
test_image_path = "mammography_images/test/Image_2.jpg"

# Preprocess the test image
test_image = preprocess_image(test_image_path)

# Make prediction
prediction = model.predict(test_image)

# Display prediction result
if prediction[0][0] > prediction[0][1]:
    print("The image is classified as Benign.")
else:
    print("The image is classified as Malignant.")


