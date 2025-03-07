from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64
from flask_cors import CORS  # Enable Cross-Origin requests

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend API calls

# Load the trained model
MODEL_PATH = "mnist_digit_recognition.h5"  # Ensure the model file is in the same directory
model = load_model(MODEL_PATH)

def preprocess_image(image):
    """
    Preprocesses input image:
    - Converts to grayscale
    - Resizes to 28x28
    - Inverts colors
    - Normalizes pixel values
    """
    image = image.convert('L').resize((28, 28))  # Convert to grayscale and resize
    image = Image.eval(image, lambda x: 255 - x)  # Invert colors (white background)
    img_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255  # Normalize
    return img_array

@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' exists

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove base64 header
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_digit = int(np.argmax(prediction))

        return jsonify({'prediction': predicted_digit})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Use debug=False for production
