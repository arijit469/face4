from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Load the trained CNN model
try:
    cnn_model = tf.keras.models.load_model('models/cnn_face_detector.h5')
    logging.info("CNN model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load CNN model: {str(e)}")
    cnn_model = None

# Load the OpenCV Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/detect_faces_cnn', methods=['POST'])
def detect_faces_cnn():
    if cnn_model is None:
        logging.error("CNN model not loaded")
        return jsonify({'error': 'CNN model not loaded'}), 500

    try:
        file = request.files['image']
        if not file:
            logging.warning("No image file provided")
            return jsonify({'error': 'No image file provided'}), 400

        img_bytes = file.read()
        if not img_bytes:
            logging.warning("Empty image file")
            return jsonify({'error': 'Empty image file'}), 400

        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logging.warning("Failed to decode image")
            return jsonify({'error': 'Failed to decode image'}), 400

        logging.info(f"Image shape: {img.shape}")

        img_resized = cv2.resize(img, (64, 64))
        img_normalized = img_resized / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        prediction = cnn_model.predict(img_input)
        confidence = float(prediction[0][0])
        
        logging.info(f"Face detection confidence: {confidence:.4f}")

        threshold = 0.5
        if confidence > threshold:
            logging.info(f"Face detected with {confidence:.2%} confidence")
            return jsonify({
                'face_detected': True,
                'confidence': confidence,
                'bounding_box': {'x': 0, 'y': 0, 'width': img.shape[1], 'height': img.shape[0]}
            })
        else:
            logging.info(f"No face detected. Confidence: {confidence:.2%}")
            return jsonify({
                'face_detected': False,
                'confidence': confidence
            })

    except Exception as e:
        logging.error(f"An error occurred during CNN face detection: {str(e)}")
        return jsonify({'error': 'An internal server error occurred'}), 500

@app.route('/detect_faces_opencv', methods=['POST'])
def detect_faces_opencv():
    try:
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            logging.info(f"OpenCV detected {len(faces)} face(s)")
            face_list = [
                {
                    'face_detected': True,
                    'bounding_box': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                } for (x, y, w, h) in faces
            ]
            return jsonify(face_list)
        else:
            logging.info("OpenCV detected no faces")
            return jsonify({'face_detected': False})

    except Exception as e:
        logging.error(f"An error occurred in OpenCV detection: {str(e)}")
        return jsonify({'error': 'An internal server error occurred'}), 500

if __name__ == '__main__':
    logging.info("Starting Face Detection Server")
    if cnn_model:
        logging.info("CNN Model loaded and ready")
    app.run(debug=True)

print("Face detection server is running. Use the /detect_faces_cnn or /detect_faces_opencv endpoints to detect faces.")