import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

img = Image.open('../ressources/anakin2.jpg')

img_array = np.array(img).astype('float32') / 255.0

img_input = np.expand_dims(img_array, axis=0)

models = ['../models/cat4.keras']

for model_path in models:
    try:
        
        model = keras.models.load_model(model_path)
        
        predictions = model.predict(img_input, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        
        print(f"\nPrediction: {class_names[predicted_class]}")
        print(f"Confidence: {confidence:.2f}%")
        
        cat_confidence = predictions[0][3] * 100 
        if predicted_class == 3:
            print(f"\nMODEL RECOGNIZED THE CAT!")
        else:
            print(f"\nModel predicted {class_names[predicted_class]} instead of Cat")
    
    except FileNotFoundError:
        print(f"\n Model not found: {model_path}")
    except Exception as e:
        print(f"\n Error with {model_path}: {e}")
