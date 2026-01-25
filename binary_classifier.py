import tensorflow as tf
from tensorflow import keras
import numpy as np

# Charger les données MNIST
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normaliser les données
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# si 0 vait s1 sinon 0
y_train_binary = (y_train == 0).astype('float32')
y_test_binary = (y_test == 0).astype('float32')

#on a une seul neuronne pour reconnaître notre problème binaire
model = keras.Sequential([
    keras.layers.Dense(1, activation='sigmoid', input_shape=(784,))
])

#descente de gradient + crossentropy
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(x_train, y_train_binary, 
                    epochs=20, batch_size=100, 
                    validation_split=0.2, verbose=1)

# Evaluer le test sue notre corpus de test
test_loss, test_accuracy = model.evaluate(x_test, y_test_binary, verbose=0)
print(f"Exactitude du test : {test_accuracy:.4f}")

# Afficher les poids appris
weights = model.get_weights()[0]
bias = model.get_weights()[1]
print(f"\nNombre de poids : {weights.shape[0]}")
print(f"Biais : {bias[0]:.4f}")

#on save dans model
model.save('models/binary_classifier.keras')