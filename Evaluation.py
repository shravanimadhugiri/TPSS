#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('improved_model.h5')

# Path to the new image you want to test
image_path = 'C:/Users/Kavana K/TPSS MODEL 1/Dataset_Train/Negative Patch/IMG_20131003_102940.jpg'  # Replace with the path to your image

# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))  # Resize to 224x224, as expected by the model
img_array = image.img_to_array(img)  # Convert image to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize the image (assuming you normalized during training)

# Make a prediction
predictions = model.predict(img_array)

# Output the prediction
class_labels = ['IR Patch', 'Negative Patch', 'Positive Patch']  # Make sure these are in the correct order
predicted_class = np.argmax(predictions)  # Get the index of the highest prediction

print(f"Predicted Class: {class_labels[predicted_class]}")
print(f"Confidence Scores: {predictions}")

# Optionally, display the image
plt.imshow(img)
plt.title(f"Predicted: {class_labels[predicted_class]}")
plt.axis('off')
plt.show()


# In[ ]:




