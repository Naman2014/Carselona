import cv2
import os
import argparse
from ultralytics import YOLO
from paddleocr import PaddleOCR

class LicensePlateRecognizer:
    def __init__(self, model_path='l_plate.pt'):
        """Initialize with YOLO and PaddleOCR models"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model file not found: {model_path}")
        
        self.model = YOLO(os.path.abspath(model_path))
        self.ocr = PaddleOCR(lang='en', show_log=False)

    def recognize(self, image):
        """
        Process image to recognize license plate text
        :param image: Input image as numpy array (BGR format)
        :return: Recognized license plate text or error message
        """
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_img = cv2.resize(processed_img, (640, 640))

        results = self.model.predict(processed_img, conf=0.3, imgsz=640, verbose=False)
        result = results[0]
        cropped_img = None

        for box in result.boxes:
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            x1, y1, x2, y2 = cords
            cropped_img = processed_img[y1:y2, x1:x2]
            break

        if cropped_img is None or cropped_img.size == 0:
            return "No license plate detected"

        try:
            ocr_results = self.ocr.ocr(cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR), cls=True)
            
            if not ocr_results or not ocr_results[0]:
                return "No text detected in license plate"
            
            license_plate_texts = [line[1][0] for line in ocr_results[0]]
            return ' '.join(license_plate_texts).upper()
        
        except Exception as e:
            return f"OCR processing error: {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='License Plate Recognition')
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to input image file')
    parser.add_argument('--model', type=str, default='l_plate.pt',
                       help='Path to YOLO model (default: l_plate.pt)')
    
    args = parser.parse_args()

    try:
        recognizer = LicensePlateRecognizer(args.model)
        
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image file not found: {args.image}")
            
        image = cv2.imread(args.image)
        if image is None:
            raise ValueError(f"Failed to read image from: {args.image}")
        
        plate_text = recognizer.recognize(image)
        print("Recognized License Plate Text:", plate_text)
    
    except Exception as e:
        print(f"Error: {str(e)}")