import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Data Loader
x = np.load('observations.npy', allow_pickle=True)
y = np.load('actions.npy', allow_pickle=True)

num_images = x.shape[0]
rows = 5
cols = 7

plt.figure(figsize=(10, 5))

# Visualize data
for i in range(rows * cols):
    if i >= num_images:
        break
    plt.subplot(rows, cols, i + 1)
    plt.imshow(x[i] * 100, cmap='gray')
    plt.title(f'Action: {y[i]}')
    plt.axis('off')
plt.show()

print(f'X Shape: {x.shape}')
print(f'Y Shape: {y.shape}')

x = np.stack(x)
y = np.array(y)
x = x / 255.0

print(x.shape[1:])

# Model
model = Sequential([
    Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=x.shape[1:]),
    Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x, y, epochs=500, batch_size=32, validation_split=0.2)

# Plot Loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Serialization
model.save('minigrid_model.h5')
