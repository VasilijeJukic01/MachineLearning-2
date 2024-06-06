import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers

# Data Loader
data = pd.read_csv('../data/crop.csv')
print(data.info())

X = data.drop('Crop', axis=1)
y = data['Crop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = y_train.nunique()

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Layer Creator
def create_layer(input_size, output_size, activation):
    return {
        'weights':      tf.Variable(tf.random.normal([input_size, output_size], dtype=tf.float64)),
        'biases':       tf.Variable(tf.random.normal([output_size], dtype=tf.float64)),
        'activation':   activation,
        'regularizer':  regularizers.l2(0.01)
    }

# NN Model
class NeuralNetwork:
    def __init__(self, input_size, output_size, eta=0.001, epochs=200, batch_size=256):
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_layer_size = input_size
        self.first_hidden_layer_size = 256
        self.second_hidden_layer_size = 256
        self.output_layer_size = output_size

        self.layers = [
            create_layer(self.input_layer_size, self.first_hidden_layer_size, tf.nn.relu),
            create_layer(self.first_hidden_layer_size, self.second_hidden_layer_size, tf.nn.relu),
            create_layer(self.second_hidden_layer_size, self.output_layer_size, tf.nn.softmax)
        ]

        self.opt = tf.keras.optimizers.Adam(learning_rate=eta)

    def train(self, training_data, training_labels):
        sample = len(training_labels)

        for epoch in range(self.epochs):
            epoch_loss = 0
            batches = int(sample / self.batch_size)

            for i in range(batches):
                batch_data = training_data[i * self.batch_size: (i + 1) * self.batch_size]
                batch_labels = training_labels[i * self.batch_size: (i + 1) * self.batch_size]
                batch_labels_onehot = tf.one_hot(batch_labels, self.output_layer_size)

                with tf.GradientTape() as tape:
                    _, output_layer_values = self.evaluate(batch_data)
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer_values, labels=batch_labels_onehot))

                    reg_loss = sum([layer['regularizer'](layer['weights']) for layer in self.layers])
                    loss = loss + reg_loss

                # Backwards
                gradients = tape.gradient(loss, [layer['weights'] for layer in self.layers] + [layer['biases'] for layer in self.layers])
                self.opt.apply_gradients(zip(gradients, [layer['weights'] for layer in self.layers] + [layer['biases'] for layer in self.layers]))

                epoch_loss += loss

            epoch_loss /= sample

            if (epoch + 1) % 5 == 0:
                print(f'Epoch: {epoch + 1}/{self.epochs}| Avg loss: {epoch_loss:.5f}')

    def evaluate(self, input_data):
        layer_input = input_data

        for layer in self.layers:
            weighted_sum = tf.add(tf.matmul(layer_input, layer['weights']), layer['biases'])
            layer_output = layer['activation'](weighted_sum)
            layer_input = layer_output

        predictions = tf.argmax(layer_output, 1)
        return predictions, weighted_sum


# Train
nn = NeuralNetwork(X_train.shape[1], num_classes, epochs=1000)
nn.train(X_train, y_train)

# Evaluate
pred, _ = nn.evaluate(X_test)
pred_correct = tf.equal(pred, y_test)
accuracy = tf.reduce_mean(tf.cast(pred_correct, tf.float32))

print(f'Model accuracy: {accuracy:.3f}')
