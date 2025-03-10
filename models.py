import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple, List

class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 18),
            nn.ReLU(),
            nn.Linear(18, 18),
            nn.ReLU(),
            nn.Linear(18, 2)
        )
        self._init_weights()

    def _init_weights(self):
        # Initialize weights using Xavier uniform and biases to zero.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x is expected to be shape [batch_size, 3, 3] (flattened inside).
        return self.net(x.view(-1, 9))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Updated CNN for better overfitting on 3x3 grids:
        # - Increase number of filters.
        # - Remove dropout and batch normalization.
        # - Use two convolutional paths to capture local and global patterns.
        self.conv1 = nn.Conv2d(1, 8, kernel_size=2, stride=1, padding=0)  # Output: (8, 2, 2)
        self.conv2 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0)  # Output: (8, 1, 1)
        # Fully connected layers: Concatenate the two paths.
        self.fc1 = nn.Linear(8*2*2 + 8*1*1, 20)  # 32+8 = 40 input features
        self.fc2 = nn.Linear(20, 2)
        self._init_weights()

    def _init_weights(self):
        # Initialize weights using Xavier uniform and biases to zero.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Ensure x is shape [batch_size, 1, 3, 3]
        x1 = torch.relu(self.conv1(x))  # shape: [batch, 8, 2, 2]
        x1 = x1.view(x1.size(0), -1)
        x2 = torch.relu(self.conv2(x))  # shape: [batch, 8, 1, 1]
        x2 = x2.view(x2.size(0), -1)
        x_combined = torch.cat([x1, x2], dim=1)
        x_combined = torch.relu(self.fc1(x_combined))
        out = self.fc2(x_combined)
        return out

def train_model(model, train_data, val_data, epochs=300, lr=0.001, weight_decay=0.01, return_history=False):
    # Extract data
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Use cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    
    # Set up early stopping
    best_val_acc = 0
    patience = 100
    patience_counter = 0
    
    # Track training history
    train_acc_history = []
    val_acc_history = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        if isinstance(model, CNN):
            outputs = model(torch.FloatTensor(X_train).reshape(-1, 1, 3, 3))
        else:
            outputs = model(torch.FloatTensor(X_train))
            
        loss = criterion(outputs, y_train.long())
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_accuracy = (predicted == y_train).sum().item() / y_train.size(0)
        train_acc_history.append(train_accuracy)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            if isinstance(model, CNN):
                val_outputs = model(torch.FloatTensor(X_val).reshape(-1, 1, 3, 3))
            else:
                val_outputs = model(torch.FloatTensor(X_val))
                
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_accuracy = (val_predicted == y_val).sum().item() / y_val.size(0)
            val_acc_history.append(val_accuracy)
        
        # Early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if return_history:
        return model, (train_acc_history, val_acc_history)
    else:
        return model, val_acc_history

def train_tree(X_train: torch.Tensor, y_train: torch.Tensor) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier for the Boolean function.
    """
    clf = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    clf.fit(X_train.view(-1, 9), y_train)
    return clf
