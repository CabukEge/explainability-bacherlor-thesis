import os
import glob
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import explainability_toolkit as ex
# Installiere das `graphviz` Paket
try:
    from graphviz import Digraph
except ImportError:
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "graphviz"])
    from graphviz import Digraph

plt.switch_backend('agg')

ex.generate_disjoint_datasets(ex.)

# Daten und Labels zusammenfügen
X = np.array(images_class0 + images_class1)
y = np.array(labels_class0 + labels_class1)


# Fully Connected Network
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
fcn_losses, fcn_models = train_fully_connected(X_tensor, y_tensor, seedList)

# Convolutional Neural Network
cnn_losses, cnn_models = train_cnn(X_tensor, y_tensor, seedList)

# Plotten der Trainingsverluste für alle Seeds
plt.figure(figsize=(10, 5))
for idx, train_losses in enumerate(fcn_losses):
    plt.plot(train_losses, label=f'FCN Seed {seedList[idx]}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('FCN Training Loss over Epochs for Different Seeds')
plt.legend()
plt.savefig('fcn_training_loss.png')
plt.close()

plt.figure(figsize=(10, 5))
for idx, train_losses in enumerate(cnn_losses):
    plt.plot(train_losses, label=f'CNN Seed {seedList[idx]}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Training Loss over Epochs for Different Seeds')
plt.legend()
plt.savefig('cnn_training_loss.png')
plt.close()

# Plotten der Gewichte der voll verbundenen Schichten (FCN)
for idx, model in enumerate(fcn_models):
    fc1_weights = model.fc1.weight.data.numpy()
    plt.figure(figsize=(10, 5))
    for i in range(fc1_weights.shape[0]):
        plt.subplot(2, 5, i+1)
        plt.imshow(fc1_weights[i].reshape(3, 3), cmap='viridis')
        plt.colorbar()
        plt.title(f'Neuron {i}')
    plt.suptitle(f'FCN Seed {seedList[idx]}: Weights of First Layer')
    plt.savefig(f'fcn_seed_{seedList[idx]}_weights.png')
    plt.close()

# Plotten der Gewichte der Convolution-Schicht (CNN)
for idx, model in enumerate(cnn_models):
    conv1_weights = model.conv1.weight.data.numpy()
    plt.figure(figsize=(10, 5))
    for i in range(conv1_weights.shape[0]):
        plt.subplot(1, 4, i+1)
        plt.imshow(conv1_weights[i, 0], cmap='viridis')
        plt.colorbar()
        plt.title(f'Filter {i}')
    plt.suptitle(f'CNN Seed {seedList[idx]}: Weights of Conv Layer')
    plt.savefig(f'cnn_seed_{seedList[idx]}_weights.png')
    plt.close()

# Visualisierung der Fully Connected Networks
def visualize_fcn(model, filename='fcn_graph'):
    dot = Digraph()

    # Input layer
    dot.node('input', 'Input Layer (3x3)')

    # Hidden layer
    for i in range(9):
        weights = model.fc1.weight.data[:, i].numpy()
        weights_str = ', '.join([f'{w:.2f}' for w in weights])
        dot.node(f'hidden_{i}', f'Neuron {i}\n({weights_str})')

    # Output layer
    dot.node('output_0', 'Output 0')
    dot.node('output_1', 'Output 1')

    # Connections
    for i in range(10):
        dot.edge('input', f'hidden_{i}')
        dot.edge(f'hidden_{i}', 'output_0')
        dot.edge(f'hidden_{i}', 'output_1')

    dot.render(filename, format='png', cleanup=True)

# Beispielaufruf zur Visualisierung des Fully Connected Network (FCN)
for idx, model in enumerate(fcn_models):
    visualize_fcn(model, f'fcn_seed_{seedList[idx]}_graph')
