import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 test data
cifar10 = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize
x_test = x_test.astype('float32') / 255.0

# Flatten labels
y_test = y_test.flatten()

# Class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print(f"Test set shape: {x_test.shape}")
print(f"Number of test images: {len(x_test)}")

# Load the model
model = keras.models.load_model('../models/cat3.keras')
print("\nModel loaded successfully!")

# Evaluate on test set
print("\nEvaluating on test set...")
y_test_categorical = keras.utils.to_categorical(y_test, 10)
test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Get predictions on test set
predictions = model.predict(x_test, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, predicted_classes)
print("\nClassification Report:")
print(classification_report(y_test, predicted_classes, target_names=class_names))

# Show some predictions
print("\n" + "="*60)
print("Sample predictions:")
print("="*60)
for i in range(10):
    true_class = class_names[y_test[i]]
    pred_class = class_names[predicted_classes[i]]
    confidence = np.max(predictions[i]) * 100
    status = "✓" if y_test[i] == predicted_classes[i] else "✗"
    print(f"{status} Image {i}: True={true_class}, Predicted={pred_class} ({confidence:.1f}%)")

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - CIFAR-10 Test Set')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('ressources/confusion_matrix2.png')
print("\nConfusion matrix saved to ressources/confusion_matrix2.png")
plt.show()