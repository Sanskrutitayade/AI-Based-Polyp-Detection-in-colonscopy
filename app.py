from flask import Flask, render_template, request, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = load_model('D:/final year flask/app/modules/plant_disease_model.h5')

# Define the class names
class_names = ['cancerous', 'non_cancerous']  # Replace with your actual class names

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to predict disease using the loaded model
def predict_disease(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    class_name = class_names[predicted_label]
    return class_name

# Flask Route to render home.html for image upload
@app.route('/')
def home():
    return render_template('home.html')

# Flask Route to handle image upload and predict disease
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image_file']

        # Save the uploaded image temporarily and predict
        filename = 'uploaded_image.jpg'
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

        # Predict disease
        predicted_class = predict_disease(image_path)

        # Pass the predicted class to the result template
        return render_template('result.html', predicted_class=predicted_class, image_name=filename)

# Endpoint to serve the uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
