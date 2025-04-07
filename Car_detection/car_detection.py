import cv2
import numpy as np
import argparse
import os
from ultralytics import YOLO

class CarDetector:
    def __init__(self, model_path='car_detection.pt'):
        """Initialize car detection model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = YOLO(os.path.abspath(model_path))

    def detect(self, image_path):
        """
        Detect and segment car in image
        :param image_path: Path to input image
        :return: Tuple (detection_status, segmented_image)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        H, W, _ = img.shape
        total_pixels = H * W

        # Perform detection
        results = self.model(img)
        max_mask = None
        max_area = 0

        for result in results:
            if result.masks:
                for mask in result.masks.data:
                    mask_np = mask.cpu().numpy() * 255
                    mask_resized = cv2.resize(mask_np, (W, H))
                    area = np.sum(mask_resized > 0)

                    if area > total_pixels * 0.3 and area > max_area:
                        max_area = area
                        max_mask = mask_resized

        if max_mask is not None:
            mask = cv2.resize(max_mask, (W, H)).astype(np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            return True, cv2.bitwise_and(img, mask)
        return False, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Car Detection')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image file')
    parser.add_argument('--model', type=str, default='car_detection.pt',
                       help='Path to car detection model (default: car_detection.pt)')
    
    args = parser.parse_args()

    try:
        detector = CarDetector(args.model)
        detected, result_img = detector.detect(args.image)
        
        if detected:
            print("Successfully detected car")
            # To save the result: cv2.imwrite('output.jpg', result_img)
        else:
            print("No car detected")
            
    except Exception as e:
        print(f"Error: {str(e)}")