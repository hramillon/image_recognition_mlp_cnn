import tensorflow as tf
from tensorflow import keras
import numpy as np

# Charger les données MNIST
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normaliser les données
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# One-hot encoding vecteur taille 10
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Créer le modèle avec 10 neuronnes, softmax et 784 entrées (pixels image)
model = keras.Sequential([
    keras.layers.Dense(10, activation='softmax', input_shape=(784,))
])

# Compiler le modèle avec crossentropy loss et descente de gradient
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

batch_size = 100
model.fit(x_train, y_train, epochs=10, batch_size=batch_size, verbose=1)

# Évaluer sur les données de test
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Exactitude du test : {test_accuracy:.4f}")

#on save dans model
model.save('models/classifier_one.keras')