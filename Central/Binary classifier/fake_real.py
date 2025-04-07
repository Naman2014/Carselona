from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf
import argparse
import os

class ImageAuthenticator:
    def __init__(self, model_path='fake_real_threshold.h5'):
        """Initialize authenticator with pre-trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = load_model(os.path.abspath(model_path))

    def authenticate_image(self, image_path):
        """Authenticate an image file"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        try:
            img = Image.open(image_path)
            img = img.resize((250, 250))
            img_arr = image.img_to_array(img)
            img_arr = np.expand_dims(img_arr, axis=0)
            img_arr = preprocess_input(img_arr)

            prediction = self.model.predict(img_arr)
            probabilities = tf.nn.softmax(prediction, axis=1)
            
            fake_prob = probabilities[0][0].numpy()
            real_prob = probabilities[0][1].numpy()

            return (True, real_prob) if real_prob > fake_prob else (False, fake_prob)
            
        except Exception as e:
            raise RuntimeError(f"Authentication failed: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Authentication')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image file')
    parser.add_argument('--model', type=str, default='fake_real_threshold.h5',
                       help='Path to authentication model (default: fake_real_threshold.h5)')
    
    args = parser.parse_args()

    try:
        authenticator = ImageAuthenticator(args.model)
        is_real, probability = authenticator.authenticate_image(args.image)
        
        result_type = "Real" if is_real else "Fake"
        print(f"Result: {result_type}")
        print(f"Confidence: {probability:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")