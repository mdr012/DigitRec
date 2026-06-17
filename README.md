# DigitRec - Handwritten Digit Recognition

A deep learning project for recognizing handwritten digits (0-9) using Convolutional Neural Networks (CNNs) built with TensorFlow/Keras.

## Features

- **Two trained models**: MNIST-based and EMNIST-based digit recognition
- **Data augmentation**: Rotation, zoom, shift, and shear for better generalization
- **Batch normalization & dropout**: For improved accuracy and reduced overfitting
- **Learning rate scheduling**: Adaptive learning rate with `ReduceLROnPlateau`
- **Interactive web app**: Streamlit interface for real-time predictions

## Model Architectures

### MNIST Model (`scripts/train_mnist.py`)
- 4 Conv2D layers (32, 32, 64, 64) with BatchNormalization
- MaxPooling, Dense(256), Dropout(0.4)
- Trained with data augmentation and EarlyStopping

### EMNIST Model (`scripts/train_emnist.py`)
- 3 Conv2D layers (32, 64, 128) with BatchNormalization
- MaxPooling, Dense(256), Dropout(0.4)
- Trained on the larger EMNIST Digits dataset (280k samples)

## Project Structure

```
DigitRec/
├── app/
│   └── app.py                 # Streamlit web interface
├── scripts/
│   ├── train_mnist.py         # Train MNIST-based model
│   └── train_emnist.py        # Train EMNIST-based model
├── notebooks/
│   ├── mainRecCode.ipynb       # Original MNIST notebook
│   └── 2ndmainRecCode.ipynb    # Original EMNIST notebook
├── model/                     # Saved trained models
├── captured_images/           # Sample test images
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Train a model
```bash
python scripts/train_mnist.py    # Train on MNIST
python scripts/train_emnist.py   # Train on EMNIST
```

### Run the web app
```bash
streamlit run app/app.py
```

## Performance

| Model   | Dataset      | Validation Accuracy |
|---------|-------------|-------------------|
| MNIST   | MNIST       | ~99.5%            |
| EMNIST  | EMNIST Digits | ~99.4%            |

## Tech Stack

- Python, TensorFlow, Keras
- OpenCV, NumPy, Matplotlib
- Streamlit
- scikit-learn
