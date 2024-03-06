from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Data Augmentation
# use "ImageDataGenerator" from Keras to perform data augmentation on the fly 
# during training, introducing variability to the training data -> helps the 
# model generalize better.
datagen = ImageDataGenerator(rotation_range=20, 
                             width_shift_range=0.1, 
                             height_shift_range=0.1, 
                             shear_range=0.2, 
                             zoom_range=0.2, 
                             horizontal_flip=True)

# Define the model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(10, 10, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model with a lower learning rate and learning rate scheduler
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Early Stopping to monitor the validation loss and stop training when the 
# validation loss stops decreasing, thereby preventing overfitting.
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with data augmentation and learning rate scheduler
history = model.fit(datagen.flow(X_train_reshaped, y_train, batch_size=32),
                    steps_per_epoch=len(X_train_reshaped) / 32,
                    epochs=50, 
                    validation_data=(X_test_reshaped, y_test),
                    callbacks=[LearningRateScheduler(scheduler), early_stopping])

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

