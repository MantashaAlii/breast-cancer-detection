import warnings
# Suppress the warning related to secure coding
warnings.filterwarnings("ignore", category=UserWarning, message="Secure coding is not enabled for restorable state")

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import cv2
import pandas as pd

# Define image dimensions
image_width = 224
image_height = 224

# Define your data directory and CSV file
data_dir = "mammography_images/train"
csv_file = "mammography_images/Training_set.csv"

def load_data(data_dir, csv_file):
    df = pd.read_csv(csv_file)
    images = []
    labels = []
    for index, row in df.iterrows():
        filename = row['filename']
        label = row['label']
        image_path = os.path.join(data_dir, filename)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (image_width, image_height))
            images.append(image)
            labels.append(label)
        else:
            print(f"Error loading image: {image_path}")
    return images, labels

# Load data
images, labels = load_data(data_dir, csv_file)

# Convert labels to numerical format
label_to_index = {label: idx for idx, label in enumerate(np.unique(labels))}
labels = [label_to_index[label] for label in labels]

# Convert labels to one-hot encoding
labels = to_categorical(labels)

# Convert images to numpy array and normalize pixel values
images = np.array(images)
images = images.astype('float32') / 255.0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_to_index), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)

# Print test accuracy
print('Test accuracy:', test_acc)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Save the trained model
model.save("breast_cancer_detection_model.h5")
print("Model saved successfully!")

# Test the model on a single image
test_image_path = "mammography_images/test/Image_1.jpg"
test_image = cv2.imread(test_image_path)
test_image = cv2.resize(test_image, (image_width, image_height))
test_image = test_image.astype('float32') / 255.0
test_image = np.expand_dims(test_image, axis=0)
prediction = model.predict(test_image)
predicted_class_index = np.argmax(prediction)
predicted_class = list(label_to_index.keys())[predicted_class_index]
print(f"The image is classified as {predicted_class} cancer.")
