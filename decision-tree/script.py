import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Data Preparation
data = pd.read_csv('../data/crop.csv')
print(data.info())
print(data.head())

X = data.drop('Crop', axis=1)
y = data['Crop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Accuracy
train_accuracy = tree.score(X_train, y_train)
test_accuracy = tree.score(X_test, y_test)

print(f"Training set accuracy: {train_accuracy:.2f}")
print(f"Test set accuracy: {test_accuracy:.2f}")

# Plot
fig = plt.figure(figsize=(25, 20))
plot_tree(tree, feature_names=X.columns, class_names=encoder.classes_, filled=True)
plt.show()


def visualize_classifier(model, X, y, ax=None, cmap='tab20'):
    ax = ax or plt.gca()

    cmap = plt.get_cmap('tab20', 22)

    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap, vmin=y.min(), vmax=y.max(), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    n_classes = len(np.unique(y))
    ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5, cmap=cmap, vmin=y.min(), vmax=y.max(), zorder=1)
    ax.set(xlim=xlim, ylim=ylim)


X_demo, y_demo = make_blobs(n_samples=300, centers=22, random_state=0, cluster_std=1.0)
plt.scatter(X_demo[:, 0], X_demo[:, 1], c=y_demo, s=50, cmap='tab20')


fig, ax = plt.subplots()
visualize_classifier(DecisionTreeClassifier(), X_demo, y_demo, ax=ax, cmap='tab20')
plt.show()
