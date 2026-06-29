# ==========================================
# Brain Tumor Detection using CNN
# Author: Rifat Hussain
# ==========================================

# Import required libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ==========================================
# Dataset Paths
# ==========================================

train_dir = "dataset/Brain_Tumor_Datasets/train"
test_dir = "dataset/Brain_Tumor_Datasets/test"

print("Training folder:", train_dir)
print("Testing folder:", test_dir)

# ==========================================
# Image Preprocessing
# ==========================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# ==========================================
# Load Training Dataset
# ==========================================

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# ==========================================
# Load Testing Dataset
# ==========================================

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

print("\nDataset loaded successfully!")
print("Classes:", train_generator.class_indices)

# ==========================================
# Build CNN Model
# ==========================================

model = Sequential()

# First Convolution Layer
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

# Second Convolution Layer
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Third Convolution Layer
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten the feature maps
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128, activation='relu'))

# Dropout to reduce overfitting
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

print("\nCNN Model Created Successfully!")

# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Compiled Successfully!")

model.summary()

# ==========================================
# Train the CNN Model
# ==========================================

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=5,
    verbose=1
)

# ==========================================
# Evaluate the Model
# ==========================================

loss, accuracy = model.evaluate(test_generator)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# ==========================================
# Save the Model
# ==========================================

model.save("models/brain_tumor_model.keras")

print("\nModel saved successfully!")

# ==========================================
# Plot Accuracy and Loss
# ==========================================

plt.figure(figsize=(12,5))

# Accuracy Graph
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss Graph
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("results/training_graph.png")
plt.show()
