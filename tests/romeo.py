import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# CIFAR-10 class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load and preprocess cat image
print("Loading cat image...")
img = Image.open('../ressources/catsquare.jpg')

# Normalize to 0-1
img_array = np.array(img).astype('float32') / 255.0

# Add batch dimension for model
img_input = np.expand_dims(img_array, axis=0)

# Try both models
models = ['../models/cat4.keras', '../models/cat3.keras']

for model_path in models:
    try:
        print(f"\n{'='*60}")
        print(f"Testing with {model_path}")
        print('='*60)
        
        model = keras.models.load_model(model_path)
        
        predictions = model.predict(img_input, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        
        print(f"\nPrediction: {class_names[predicted_class]}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"\nTop 3 predictions:")
        for i, idx in enumerate(top_3_indices, 1):
            print(f"  {i}. {class_names[idx]}: {predictions[0][idx]*100:.2f}%")
        
        cat_confidence = predictions[0][3] * 100 
        if predicted_class == 3:
            print(f"\n✓ MODEL RECOGNIZED THE CAT!")
        else:
            print(f"\n✗ Model predicted {class_names[predicted_class]} instead of Cat")
            print(f"  (Cat confidence: {cat_confidence:.2f}%)")
    
    except FileNotFoundError:
        print(f"\n✗ Model not found: {model_path}")
    except Exception as e:
        print(f"\n✗ Error with {model_path}: {e}")
