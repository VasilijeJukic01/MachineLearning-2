import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

x = np.load('observations.npy', allow_pickle=True)
y = np.load('actions.npy', allow_pickle=True)

print(f'X Shape: {x.shape}')
print(f'Y Shape: {y.shape}')

x = np.stack(x)
y = np.array(y)
x = x / 255.0

model = Sequential([
    Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=x.shape[1:]),
    Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x, y, epochs=500, batch_size=32, validation_split=0.2)

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

model.save('minigrid_model.h5')
