import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model, model_from_json

# Load currency recognition model
def load_currency_model():
    json_file = open("40nepalese_currency_recognition_model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    currency_recognition_model = model_from_json(loaded_model_json)
    currency_recognition_model.load_weights("models/40nepalese_currency_recognition_model.h5")
    return currency_recognition_model

# Load fraud detection model
fraud_detection_model = load_model('models/30cnnmodel.h5')

# Function to preprocess the image before feeding it to the classification model
def preprocess_classification_image(image):
    resized_img = cv2.resize(image, (224, 224))
    img = np.expand_dims(resized_img, axis=0)
    scaled_img = img / 255.0
    return scaled_img

# Function to preprocess the image before feeding it to the detection model
def preprocess_detection_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to classify currency
def classify_currency(image):
    img = preprocess_classification_image(image)
    currencies = ['Fifty', 'Five', 'Five Hundred', 'Hundred', 'Ten', 'Thousand', 'Twenty']
    prediction = currency_recognition_model.predict(img)
    prediction_class = np.argmax(prediction)
    result = currencies[prediction_class]
    return result

# Function to detect fraud
def detect_fraud(img):
    prediction = fraud_detection_model.predict(img)
    label = int(prediction[0][0])
    if label == 0:
        return 'Fake'
    if label == 1:
        return 'Genuine'

# Streamlit web app
st.title("Currency Recognition and Fraud Detection Web App")
st.write("Upload an image to predict the currency and detect fraud:")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the uploaded image to numpy array
    image_np = np.array(image)

    # Load the currency recognition model
    currency_recognition_model = load_currency_model()

    # Predict currency
    currency_result = classify_currency(image_np)
    st.write(f"Predicted Currency: {currency_result}")

    # Preprocess image for fraud detection
    fraud_detection_img = preprocess_detection_image(uploaded_file)

    # Predict fraud
    fraud_result = detect_fraud(fraud_detection_img)
    st.write(f"Fraud Detection: {fraud_result}")





