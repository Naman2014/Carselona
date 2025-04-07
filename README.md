This toolkit has three folders that fulfil different needs:
- Binary classifier : it has two scripts fake_real.py and door_status.py 
- Car detection : it has a script car_detection.py 
- License plate : contains script that reads license plate


This toolkit provides four standalone Python modules that utilize deep learning to perform various image analysis tasks including:
- Door status classification
- Image authenticity check
- Car detection
- License plate recognition

Each script is designed to be run individually with a trained model and an input image.

---

## 📁 Contents

| Script | Functionality |
|--------|---------------|
| `door_status.py` | Classifies whether a door is open or closed |
| `fake_real.py` | Detects whether an image is real or fake |
| `car_detection.py` | Detects and segments cars from an image |
| `l_plate.py` | Recognizes text from license plates using OCR |

---

## 🔧 Requirements

Install the following Python packages before running any script:

```bash
pip install -r requirements.txt
```

requirements.txt are already in each folder


> Ensure you're using a compatible version of `tensorflow` with `keras` and that YOLO models are supported by `ultralytics`.

---

## 🚪 1. Door Status Classification

**Script:** `door_status.py`  
**Model:** Trained `.h5` Keras model (default: `test_door.h5`)  
**Command:**

```bash
python door_status.py --image <path_to_image> --model test_door.h5
```

**Output:**
- Door status (`Open` or `Closed`)
- Confidence score

---

## 🖼️ 2. Fake vs Real Image Detection

**Script:** `fake_real.py`  
**Model:** Trained `.h5` Keras model (default: `fake_real_threshold.h5`)  
**Command:**

```bash
python fake_real.py --image <path_to_image> --model fake_real_threshold.h5
```

**Output:**
- Result (`Fake` or `Real`)
- Confidence score

---

## 🚗 3. Car Detection and Segmentation

**Script:** `car_detection.py`  
**Model:** YOLO model (default: `car_detection.pt`)  
**Command:**

```bash
python car_detection.py --image <path_to_image> --model car_detection.pt
```

**Output:**
- Console status of car detection
- Optionally save segmented image using `cv2.imwrite`

---

## 🔢 4. License Plate Recognition

**Script:** `l_plate.py`  
**Model:** YOLO for detection (default: `l_plate.pt`)  
**Additional:** Uses PaddleOCR for text recognition  
**Command:**

```bash
python l_plate.py --image <path_to_image> --model l_plate.pt
```

**Output:**
- Recognized license plate text
- Error message if detection/OCR fails

---

## 📌 Notes

- All scripts accept optional `--model` arguments to specify custom model paths.
- Models must be trained beforehand and compatible with the respective framework (`Keras` or `YOLO`).
- Ensure input images are readable and correctly preprocessed.
- For YOLO, the model weights must be compatible with the `ultralytics` package.

---

