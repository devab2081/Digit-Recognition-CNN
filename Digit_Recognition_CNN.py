# ===========================================
# CHARACTER & DIGIT RECOGNITION USING CNN
# ===========================================

# Import required libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# ===========================================
# 1. LOAD DATASET (MNIST for digits 0–9)
# ===========================================
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Normalize pixel values (0–1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add channel dimension for CNN (28x28x1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Training set shape:", x_train.shape)
print("Test set shape:", x_test.shape)

# ===========================================
# 2. USER INPUT FOR TRAINING SETTINGS
# ===========================================
try:
    epochs = int(input("Enter number of epochs (e.g., 2 to 10): "))
except:
    epochs = 3
    print("Invalid input. Using default epochs =", epochs)

try:
    batch_size = int(input("Enter batch size (e.g., 32 or 64): "))
except:
    batch_size = 64
    print("Invalid input. Using default batch size =", batch_size)

# ===========================================
# 3. BUILD CNN MODEL
# ===========================================
model = models.Sequential([
    # Convolution + Pooling layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten + Fully Connected layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),   # prevents overfitting
    layers.Dense(10, activation='softmax')  # output layer for 10 digits
])

# ===========================================
# 4. COMPILE MODEL
# ===========================================
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display model architecture
model.summary()

# ===========================================
# 5. TRAIN THE MODEL
# ===========================================
print(f"\nTraining the CNN model for {epochs} epochs with batch size {batch_size}...\n")
history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    verbose=2
)

# ===========================================
# 6. EVALUATE PERFORMANCE
# ===========================================
print("\nEvaluating model on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# ===========================================
# 7. VISUALIZE ACCURACY & LOSS
# ===========================================
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ===========================================
# 8. TEST PREDICTION ON A RANDOM SAMPLE
# ===========================================
index = np.random.randint(0, len(x_test))
sample = x_test[index]
true_label = y_test[index]

plt.imshow(sample.reshape(28, 28), cmap='gray')
plt.title(f"Actual Digit: {true_label}")
plt.axis('off')
plt.show()

# Predict
pred = model.predict(sample.reshape(1, 28, 28, 1))
predicted_label = np.argmax(pred)
print(f"🔢 Predicted Digit: {predicted_label}")
