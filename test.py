import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Generate random vector data and labels
num_samples = 1000
vector_length = 100

X_train = np.random.rand(num_samples, vector_length)
X_test = np.random.rand(num_samples // 4, vector_length)  # 1/4 of the data for testing

# Generate random labels (binary classification)
y_train = np.random.randint(2, size=num_samples)
y_test = np.random.randint(2, size=num_samples // 4)

# Reshape the data into 10x10 "images" with 1 channel
X_train_reshaped = X_train.reshape(-1, 10, 10, 1)
X_test_reshaped = X_test.reshape(-1, 10, 10, 1)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(10, 10, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test))
