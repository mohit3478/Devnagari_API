import os
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    # Example preprocessing (resize and convert to numpy array)
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image_array = np.array(image)
    return image_array

if __name__ == "__main__":
    image_path = 'path_to_your_image.png'
    preprocessed_image = preprocess_image(image_path)
    print(f'Preprocessed image shape: {preprocessed_image.shape}')
