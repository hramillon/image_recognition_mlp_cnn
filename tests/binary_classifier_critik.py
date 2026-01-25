import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

model = keras.models.load_model('../models/binary_classifier.keras')

zeros_mask = (y_test == 0)
x_test_zeros = x_test[zeros_mask]

print(f"Total images in test set: {len(x_test)}")
print(f"Images that are 0: {len(x_test_zeros)}")
print(f"Percentage of 0s: {len(x_test_zeros) / len(x_test) * 100:.2f}%")
print()

predictions = model.predict(x_test_zeros)
predictions_binary = (predictions > 0.5).astype(int).flatten()

expected = np.ones(len(x_test_zeros))

accuracy_on_zeros = np.mean(predictions_binary == expected)

print(f"Accuracy on ZEROS ONLY: {accuracy_on_zeros * 100:.2f}%")
print(f"Correct predictions: {np.sum(predictions_binary == expected)} / {len(x_test_zeros)}")
print(f"Wrong predictions: {np.sum(predictions_binary != expected)}")
print()

mean_confidence_zeros = np.mean(predictions[predictions_binary == 1])
mean_confidence_wrong = np.mean(predictions[predictions_binary == 0])

print(f"Mean confidence when predicting 'is zero' correctly: {mean_confidence_zeros:.4f}")
if len(predictions[predictions_binary == 0]) > 0:
    print(f"Mean confidence when predicting WRONG: {mean_confidence_wrong:.4f}")