import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load CIFAR-10
cifar10 = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# One-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print(f"Training set shape: {x_train.shape}")
print(f"Test set shape: {x_test.shape}")
print(f"Classes: {class_names}")

# Model with Dropout and increased complexity
model = keras.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    keras.layers.Dropout(0.25),
    keras.layers.MaxPooling2D((2,2)),
    
    keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    keras.layers.Dropout(0.25),
    keras.layers.MaxPooling2D((2,2)),
    
    keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    keras.layers.Dropout(0.25),
    
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

print("\nModel Architecture:")
model.summary()

# Compile with Adam
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Class weights
class_weight = {
    0: 1.0,      # Airplane
    1: 1.0,      # Automobile
    2: 1.5,      # Bird (augmente - classe difficile)
    3: 1.8,      # Cat (augmente beaucoup - classe difficile)
    4: 1.0,      # Deer
    5: 1.5,      # Dog (augmente - classe difficile)
    6: 0.8,      # Frog (diminue)
    7: 0.8,      # Horse (diminue)
    8: 1.0,      # Ship
    9: 1.0       # Truck
}

print("\nClass weights applied:")
for cls_id, weight in class_weight.items():
    print(f"  {class_names[cls_id]}: {weight}")

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/cat4.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        verbose=1
    )
]

print("\nTraining with class weights and dropout...")
history = model.fit(x_train, y_train, 
                    epochs=50,
                    batch_size=64,
                    callbacks=callbacks,
                    validation_split=0.2,
                    class_weight=class_weight,
                    verbose=1)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

model.save('model/cat4.keras')

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CIFAR-10 Training Loss (with Dropout & Class Weights)')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CIFAR-10 Training Accuracy (with Dropout & Class Weights)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('ressources/cifar10_results.png')
print("Results plot saved to ressources/cifar10_results.png")
plt.show()