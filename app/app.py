import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras

st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognizer")
st.markdown("Upload an image of a handwritten digit (0-9) and the model will predict it.")

@st.cache_resource
def load_model():
    model_path = './model/2nddigit_emnist_model.h5'
    try:
        model = keras.models.load_model(model_path)
        return model
    except:
        try:
            alt_path = '../model/2nddigit_emnist_model.h5'
            model = keras.models.load_model(alt_path)
            return model
        except:
            try:
                alt_path2 = 'model/2nddigit_emnist_model.h5'
                model = keras.models.load_model(alt_path2)
                return model
            except:
                return None

model = load_model()

if model is None:
    st.error("No trained model found! Please train the model first using `scripts/train_emnist.py` or `scripts/train_mnist.py`.")
    st.stop()

model_choice = st.radio(
    "Select model:",
    ["EMNIST Model (recommended)", "MNIST Model"],
    horizontal=True
)

@st.cache_resource
def load_mnist_model():
    try:
        return keras.models.load_model('./model/digit_recognizer_improved.h5')
    except:
        try:
            return keras.models.load_model('../model/digit_recognizer_improved.h5')
        except:
            try:
                return keras.models.load_model('model/digit_recognizer_improved.h5')
            except:
                return None

if model_choice == "MNIST Model":
    mnist_model = load_mnist_model()
    if mnist_model is not None:
        model = mnist_model
    else:
        st.warning("MNIST model not found, falling back to EMNIST model.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    help="Upload a clear image of a single handwritten digit"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    img_array = np.array(image)

    img_resized = cv2.resize(img_array, (28, 28))
    img_inverted = 255 - img_resized
    img_normalized = img_inverted.astype('float32') / 255.0
    img_input = np.expand_dims(img_normalized, axis=(0, -1))

    pred = model.predict(img_input, verbose=0)
    digit = np.argmax(pred)
    confidence = float(np.max(pred))

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width=200, use_container_width=False)

    with col2:
        st.markdown("### Prediction")
        st.markdown(f"# **{digit}**")
        st.markdown(f"Confidence: **{confidence:.2%}**")

        if confidence > 0.9:
            st.success(f"I am {confidence:.1%} sure this is a **{digit}**")
        elif confidence > 0.7:
            st.warning(f"I think this might be a **{digit}** ({confidence:.1%} confidence)")
        else:
            st.error(f"Uncertain prediction. Could be a **{digit}** (only {confidence:.1%} confidence)")

    st.markdown("---")
    st.markdown("### Prediction Probabilities")
    proba = pred[0]
    proba_chart_data = {"Digit": list(range(10)), "Probability": proba}
    st.bar_chart(proba_chart_data, x="Digit", y="Probability", height=300)

st.markdown("---")
st.markdown(
    "*Powered by TensorFlow & Keras | Trained on MNIST & EMNIST datasets*",
    help="This app uses convolutional neural networks trained on handwritten digit datasets."
)
