import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Function to load the violence dataset (you need to adjust the path)
def load_violence_dataset(path):
    videos = []
    labels = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        print(f"Processing folder: {folder_path}")
        if os.path.isdir(folder_path):
            label = 1 if 'violence' in folder.lower() else 0  # 1: Violence, 0: Non-violence
            for file in os.listdir(folder_path):
                if file.endswith(('.jpg', '.png')):
                    image_path = os.path.join(folder_path, file)
                    try:
                        image = cv2.imread(image_path)
                        if image is None:
                            print(f"Failed to load image {image_path}. Skipping...")
                            continue

                        image = cv2.resize(image, (224, 224))  # Resize to the model input size
                        videos.append(image)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
                        continue
    videos = np.array(videos)
    labels = np.array(labels)
    return videos, labels

# Preprocess the dataset
def preprocess_data(images):
    images = images.astype('float32') / 255.0
    return images

# Autoencoder Model
def create_autoencoder(input_shape):
    # Encoder
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Bottleneck
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Decoder
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder

# Compile the model
def compile_autoencoder(autoencoder):
    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), 
                        loss='mean_squared_error',
                        metrics=['accuracy'])

# Train the autoencoder model
def train_autoencoder(autoencoder, train_images, val_images):
    batch_size = 8
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = autoencoder.fit(train_images, train_images,  # Autoencoders train to reconstruct the input
                              validation_data=(val_images, val_images),
                              epochs=10,
                              batch_size=batch_size,
                              callbacks=[early_stopping])
    return history

# Test the model and calculate performance metrics
def test_autoencoder(autoencoder, test_images, true_labels, threshold=0.02):
    reconstructions = autoencoder.predict(test_images)
    
    # Compute reconstruction error
    errors = np.mean(np.abs(reconstructions - test_images), axis=(1, 2, 3))
    anomalies = errors > threshold
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, anomalies)
    precision = precision_score(true_labels, anomalies)
    recall = recall_score(true_labels, anomalies)
    f1 = f1_score(true_labels, anomalies)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    return errors, anomalies, accuracy, precision, recall, f1

# Main function to execute the workflow
def main():
    dataset_path = r'archive\new_violence'  # Update with the actual dataset path
    
    # Load the dataset and labels
    images, labels = load_violence_dataset(dataset_path)
    images = preprocess_data(images)
    
    # Split into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Further split training data into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # Create and compile the autoencoder model
    input_shape = (224, 224, 3)
    autoencoder = create_autoencoder(input_shape)
    compile_autoencoder(autoencoder)
    
    # Train the autoencoder
    print("Training the autoencoder...")
    history = train_autoencoder(autoencoder, X_train, X_val)
    
    # Plot training history
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
    
    # Test the autoencoder and calculate accuracy, precision, recall, F1
    print("Testing the autoencoder...")
    errors, anomalies, accuracy, precision, recall, f1 = test_autoencoder(autoencoder, X_test, y_test)

if __name__ == "__main__":
    main()
