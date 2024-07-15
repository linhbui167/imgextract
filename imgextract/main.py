from flask import Flask, request, jsonify, send_file
from io import BytesIO
import cv2
import os
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from object_detection.detection import extract_painting, image_to_bytes



app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'

# Ensure the upload and extracted directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_picture_automatically(input_image_path, cascade_path='haarcascade_frontalface_default.xml'):
    """
    Automatically detect and extract the picture from an image using Haar Cascade for face detection.
    Returns the cropped image as a PIL Image object.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    image = cv2.imread(input_image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_image = image[y:y+h, x:x+w]
        cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        return cropped_pil_image
    else:
        return None

    
@app.route('/', methods=['GET'])
def index():
    return "Image Extract API!"

@app.route('/upload', methods=['POST'])
def upload_file():
    print(request.files)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        input_image_path = os.path.join(UPLOAD_FOLDER, file.filename)

        # Save the uploaded file
        file.save(input_image_path)

        # Extract the region of interest
        cropped_image = extract_painting(input_image_path)
        os.remove(input_image_path)
        if cropped_image is not None:
            img_io = BytesIO()
            cropped_image.save(img_io, 'JPEG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'No faces detected'}), 400
        
def main():
    app.run(port=8000, debug=True)

if __name__ == '__main__':
    main()
