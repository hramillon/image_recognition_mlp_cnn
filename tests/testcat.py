import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

cifar10 = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_test = x_test.astype('float32') / 255.0

y_test = y_test.flatten()

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = keras.models.load_model('../models/cat4.keras')

y_test_categorical = keras.utils.to_categorical(y_test, 10)
test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

predictions = model.predict(x_test, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, predicted_classes)
print("\nClassification Report:")
print(classification_report(y_test, predicted_classes, target_names=class_names))

print("Sample predictions:")
for i in range(10):
    true_class = class_names[y_test[i]]
    pred_class = class_names[predicted_classes[i]]
    confidence = np.max(predictions[i]) * 100
    status = "yes" if y_test[i] == predicted_classes[i] else "no"
    print(f"{status} Image {i}: True={true_class}, Predicted={pred_class} ({confidence:.1f}%)")