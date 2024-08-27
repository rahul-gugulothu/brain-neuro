import os
from flask import Flask, request, jsonify, send_from_directory
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_file_path = "C:/Users/Rahul/Desktop/New folder/vggl (2).keras"  

if not os.path.exists(model_file_path):
    raise FileNotFoundError(f"Model file not found: {model_file_path}")

try:
    saved_model = load_model(model_file_path)
    print("Model loaded successfully")
except OSError as e:
    print(f"Error loading model: {e}")
    saved_model = None

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def predict(img):
    img = img.resize((224, 224))  
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)
    output = saved_model.predict(img_array)
    if output[0][0] > output[0][1]:
        return "No"
    else:
        return "Yes"


@app.route('/', methods=['GET'])
def home():
    return "Hello, welcome"


@app.route('/predict', methods=['POST'])
def upload_file():
    print("Request received")  
    if 'image' not in request.files:
        print("No file part in the request")  
        return jsonify({'error': 'No file part in the request'}), 400

    image_data = request.files['image']

    if image_data.filename == '':
        print("No file selected for uploading")  
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_data.filename)
        image_data.save(image_path)

        img = Image.open(image_path).convert('RGB')
        prediction = predict(img)

        image_url = request.host_url + 'uploads/' + image_data.filename
        return jsonify({'prediction': prediction, 'image_url': image_url})
    except (IOError, Exception) as e:  
        print(f"Error processing the file: {e}")
        return jsonify({'error': 'Error processing the file'}), 400


@app.route('/uploads/<filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    if saved_model is not None:
        app.run(debug=True, port=5000)
    else:
        print("Failed to load model, Flask app is not running")
