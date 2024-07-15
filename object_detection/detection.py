# object_detection/detection.py

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# Load a pre-trained model (e.g., SSD MobileNet V2)
model = tf.saved_model.load("ssd_mobilenet_v2/saved_model")

def load_image_into_numpy_array(path):
    return np.array(cv2.imread(path))

def detect_objects(image_np):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis,...]

    detections = model(input_tensor)

    return detections

def visualize_detection(image_np, detections):
    image_np_with_detections = image_np.copy()

    # Visualization of the results of a detection.
    for i in range(detections['detection_boxes'].shape[0]):
        if detections['detection_scores'][i] > 0.5:
            box = tuple(detections['detection_boxes'][i].numpy())
            image_np_with_detections = cv2.rectangle(
                image_np_with_detections, 
                (int(box[1] * image_np.shape[1]), int(box[0] * image_np.shape[0])), 
                (int(box[3] * image_np.shape[1]), int(box[2] * image_np.shape[0])),
                (255, 0, 0), 2
            )
    return image_np_with_detections

def extract_painting(image_path):
    image_np = load_image_into_numpy_array(image_path)
    detections = detect_objects(image_np)
    image_np_with_detections = visualize_detection(image_np, detections)
    return image_np_with_detections

def image_to_bytes(image):
    img_io = BytesIO()
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_image.save(img_io, 'JPEG')
    img_io.seek(0)
    return img_io
