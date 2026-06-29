import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = tf.keras.models.load_model("models/brain_tumor_model.keras")

# Path of image to predict
img_path = "sample.jpg"   
# Load image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Prediction: Brain Tumor Detected")
else:
    print("Prediction: No Brain Tumor")