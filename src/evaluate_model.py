import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Constants
MODEL_PATH = '../trained_models/your_model.h5'
TEST_IMAGE_PATH = 'path/to/your/test/image.png'
CLASS_NAMES = ['class1', 'class2', ...]  # Replace with actual class names

# Load the trained model
model = load_model(MODEL_PATH)

# Load and preprocess the test image
img = image.load_img(TEST_IMAGE_PATH, target_size=(32, 32))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.0  # Normalize

# Predict the class probabilities
predictions = model.predict(x)

# Get the predicted class index and name
predicted_class_idx = np.argmax(predictions)
predicted_class_name = CLASS_NAMES[predicted_class_idx]

print(f'Predicted class: {predicted_class_name}')
