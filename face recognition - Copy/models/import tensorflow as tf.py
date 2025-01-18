import tensorflow as tf
from tensorflow.keras import layers, models
from torchvision import datasets, transforms
import os
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_image(image_path, label):
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [64, 64])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    except tf.errors.InvalidArgumentError as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None, None

def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def prepare_dataset(face_dir, non_face_dir):
    face_images = [os.path.join(face_dir, f) for f in os.listdir(face_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    non_face_images = [os.path.join(non_face_dir, f) for f in os.listdir(non_face_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    all_images = face_images + non_face_images
    labels = [1] * len(face_images) + [0] * len(non_face_images)
    
    dataset = tf.data.Dataset.from_tensor_slices((all_images, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda x, y: x is not None)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=len(all_images))
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def train_model(model, dataset, epochs=10):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(dataset, epochs=epochs)
    return history

def main():
    face_dir = 'D:/New folder (3)/data/face_images'
    non_face_dir = 'D:/New folder (3)/data/non_face_images'
    
    if not os.path.exists(face_dir) or not os.path.exists(non_face_dir):
        logging.error(f"Error: Directories '{face_dir}' or '{non_face_dir}' not found.")
        return
    
    logging.info("Preparing dataset...")
    dataset = prepare_dataset(face_dir, non_face_dir)
    
    logging.info("Creating CNN model...")
    model = create_cnn_model()
    
    logging.info("Training model...")
    history = train_model(model, dataset)
    
    logging.info("Saving model...")
    model.save('models/cnn_face_detector.h5')
    logging.info("Model trained and saved successfully.")

    # Print model summary
    model.summary()

if __name__ == "__main__":
    main()