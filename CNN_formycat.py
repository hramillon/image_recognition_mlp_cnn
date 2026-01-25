import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np

# On récupère cifar comme Mnist
cifar10 = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# normalisation (valeur entre 0 et 1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# on passe de dimesion (N, 1) à (N,)
y_train = y_train.flatten()
y_test = y_test.flatten()

# One-hot
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# CIFAR-10 catégories
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#info sur la taille des sets
print(f"Training set shape: {x_train.shape}")
print(f"Test set shape: {x_test.shape}")
print(f"Classes: {class_names}")

# On adapte le modèle qu'on a fait pour Mnist
model = keras.Sequential([
    keras.layers.Conv2D(96, (3,3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2,2)),
    
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2,2)),
    
    keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2,2)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#pour avoir une idée du modèle en cas d'utilisation ext
print("\nModel Architecture:")
model.summary()

#entraine
batch_size = 100
history = model.fit(x_train, y_train, 
                    epochs=20, 
                    batch_size=batch_size, 
                    validation_split=0.2,
                    verbose=1)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

model.save('models/cat1_like_mnist.keras')

# Plot training history

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CIFAR-10 Training Loss')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CIFAR-10 Training Accuracy')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('images/cifar10_results.png')
print("Results plot saved to images/cifar10_results.png")
plt.show()