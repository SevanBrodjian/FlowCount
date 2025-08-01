import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers, models

def load_data(video_path, label_path):
    """
    Load video frames and corresponding labels for training.

    Parameters:
    -----------
    video_path : str
        Path to the video file.
    label_path : str
        Path to the label file.

    Returns:
    --------
    frame_pairs : numpy.ndarray
        Array of consecutive frame pairs.
    labels : numpy.ndarray
        Array of total displacement labels.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frames = np.array(frames)

    # Create pairs of consecutive frames
    frame_pairs = np.array([frames[i:i+2] for i in range(len(frames)-1)])

    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            labels.append(float(parts[4]))  # Assuming the fifth column is the total displacement
    labels = np.array(labels)

    return frame_pairs, labels

def create_model(input_shape):
    """
    Create a deep learning model for displacement estimation.

    Parameters:
    -----------
    input_shape : tuple
        Shape of the input data.

    Returns:
    --------
    model : tensorflow.keras.Model
        Compiled model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_counting_model(data_path, model_save_path):
    """
    Train a counting model using the provided data.

    Parameters:
    -----------
    data_path : str
        Path to the directory containing the training data.
    model_save_path : str
        Path where the trained model will be saved.
    """
    video_path = os.path.join(data_path, 'drifting_dots_X.mp4')
    label_path = os.path.join(data_path, 'drifting_dots_Y.txt')

    frame_pairs, labels = load_data(video_path, label_path)
    input_shape = frame_pairs[0].shape

    model = create_model(input_shape)
    model.fit(frame_pairs, labels, epochs=10, batch_size=32, validation_split=0.2)

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    # Example usage
    train_counting_model(data_path='./data', model_save_path='./model') 