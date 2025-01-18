import tensorflow as tf
from tensorflow.keras import layers, models
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load and preprocess image
# CelebFaces Attributes (CelebA) Dataset


def load_and_preprocess_image(image_path, label):
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [64, 64])  # Resize to 64x64
        img = tf.cast(img, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
        return img, label
    except tf.errors.InvalidArgumentError as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None, None

# Function to create CNN model
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    return model

# Function to prepare the dataset
def prepare_dataset(face_dir, non_face_dir):
    # Get paths of face and non-face images
    face_images = [os.path.join(face_dir, f) for f in os.listdir(face_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    non_face_images = [os.path.join(non_face_dir, f) for f in os.listdir(non_face_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Combine images and assign labels (1 for faces, 0 for non-faces)
    all_images = face_images + non_face_images
    labels = [1] * len(face_images) + [0] * len(non_face_images)
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((all_images, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda x, y: x is not None)  # Remove None values from the dataset
    dataset = dataset.cache()  # Cache the dataset for performance
    dataset = dataset.shuffle(buffer_size=len(all_images))  # Shuffle dataset
    dataset = dataset.batch(32)  # Set batch size to 32
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch for better performance
    
    return dataset

# Function to train the model
def train_model(model, dataset, epochs=10):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Binary cross-entropy for binary classification
                  metrics=['accuracy'])
    
    history = model.fit(dataset, epochs=epochs)
    return history

# Main function to run the script
def main():
    # Directories where your images are stored
    face_dir = 'D:/New folder (3)/face recognition/data/face_images'
    non_face_dir = 'D:/New folder (3)/face recognition/data/non_face_images'
    
    # Check if directories exist
    if not os.path.exists(face_dir) or not os.path.exists(non_face_dir):
        logging.error(f"Error: Directories '{face_dir}' or '{non_face_dir}' not found.")
        return
    
    # Prepare the dataset
    logging.info("Preparing dataset...")
    dataset = prepare_dataset(face_dir, non_face_dir)
    
    # Create CNN model
    logging.info("Creating CNN model...")
    model = create_cnn_model()
    
    # Train the model
    logging.info("Training model...")
    history = train_model(model, dataset)
    
    # Save the trained model
    logging.info("Saving model...")
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/cnn_face_detector.h5')
    logging.info("Model trained and saved successfully.")

    # Print model summary
    model.summary()

# Run the script
if __name__ == "__main__":
    main()
