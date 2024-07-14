from flask import Flask, request, jsonify
import cv2
import os
from PIL import Image
import numpy as np

app = Flask(__name__)

def extract_picture_automatically(input_image_path, output_image_path, cascade_path='haarcascade_frontalface_default.xml'):
    """
    Automatically detect and extract the picture from an image using Haar Cascade for face detection.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    image = cv2.imread(input_image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_image = image[y:y+h, x:x+w]
        cv2.imwrite(output_image_path, cropped_image)
        return True
    else:
        return False
    
@app.route('/', methods=['GET'])
def index():
    return "Welcome to the Image Extract API!"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        input_image_path = os.path.join('uploads', file.filename)
        output_image_path = os.path.join('extracted', file.filename)

        # Save the uploaded file
        file.save(input_image_path)

        # Create directories if they don't exist``
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

        # Extract the region of interest
        if extract_picture_automatically(input_image_path, output_image_path):
            return jsonify({'message': 'Image processed', 'output_image_path': output_image_path}), 200
        else:
            return jsonify({'error': 'No faces detected'}), 400
        
def main():
    app.run(debug=True)

if __name__ == '__main__':
    main()
