from keras.preprocessing import image as keras_image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf
import argparse
import os

class DoorStatusClassifier:
    def __init__(self, model_path='test_door.h5'):
        """Initialize door status classifier"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = load_model(os.path.abspath(model_path))

    def classify_door(self, image_path):
        """Classify door status from image file"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        try:
            img = Image.open(image_path)
            img = img.resize((250, 250))
            img_arr = keras_image.img_to_array(img)
            img_arr = np.expand_dims(img_arr, axis=0)
            img_arr = preprocess_input(img_arr)

            prediction = self.model.predict(img_arr)
            probabilities = tf.nn.softmax(prediction, axis=1)
            
            closed_prob = probabilities[0][0].numpy()
            open_prob = probabilities[0][1].numpy()

            return (open_prob > closed_prob), max(open_prob, closed_prob)
            
        except Exception as e:
            raise RuntimeError(f"Classification failed: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Door Status Classification')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image file')
    parser.add_argument('--model', type=str, default='test_door.h5',
                       help='Path to classification model (default: test_door.h5)')
    
    args = parser.parse_args()

    try:
        classifier = DoorStatusClassifier(args.model)
        is_open, confidence = classifier.classify_door(args.image)
        
        status = "Open" if is_open else "Closed"
        print(f"Door Status: {status}")
        print(f"Confidence: {confidence:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")