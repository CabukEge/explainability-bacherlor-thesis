import models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

SEED_LIST = [42, 18215, 14564, 74079, 24555, 60045, 3, 58064034, 25190, 34988]
ST_NUM_EPOCHS = 500


# Seed setzen für Reproduzierbarkeit
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Sklearn Entscheidungsbaum
def decision_tree(X, y):
    X_flat = X.reshape(len(X), -1)
    clf = DecisionTreeClassifier()
    clf.fit(X_flat, y)
    y_pred = clf.predict(X_flat)
    accuracy = accuracy_score(y, y_pred)
    print(f'Decision Tree Accuracy: {accuracy:.4f}')


def train_fully_connected(X, y, seedList, num_epochs):
    all_train_losses = []
    final_models = []
    for i in seedList:
        models.set_seed(i)
        model = models.SimpleNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        train_losses = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if (epoch+1) % 100 == 0:
                print(f'FCN Seed {i}: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        all_train_losses.append(train_losses)
        final_models.append(model)
        with torch.no_grad():
            predicted = model(X).argmax(dim=1)
            accuracy = (predicted == y).float().mean()
            print(f'FCN Seed {i}: Accuracy: {accuracy:.4f}')
            print(f'FCN Seed {i}: Predicted labels:', predicted.numpy())
            print(f'FCN Seed {i}: True labels:', y.numpy())
    return all_train_losses, final_models


def train_cnn(X, y, seedList, num_epochs):
    all_train_losses = []
    final_models = []
    X = X.unsqueeze(1)  # Hinzufügen einer Kanal-Dimension
    for i in seedList:
        models.set_seed(i)
        model = models.SimpleCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        train_losses = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if (epoch+1) % 100 == 0:
                print(f'CNN Seed {i}: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        all_train_losses.append(train_losses)
        final_models.append(model)
        with torch.no_grad():
            predicted = model(X).argmax(dim=1)
            accuracy = (predicted == y).float().mean()
            print(f'CNN Seed {i}: Accuracy: {accuracy:.4f}')
            print(f'CNN Seed {i}: Predicted labels:', predicted.numpy())
            print(f'CNN Seed {i}: True labels:', y.numpy())
    return all_train_losses, final_models