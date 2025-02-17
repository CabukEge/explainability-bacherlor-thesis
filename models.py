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
        # First convolution path - for local patterns
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=1, padding=0)
        # Second convolution path - for global patterns
        self.conv2 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0)
        # Fully connected layers
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 2)
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(4)
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Ensure x is shape [batch_size, 1, 3, 3]
        path1 = self.conv1(x)
        path1 = self.bn1(path1)
        path1 = torch.relu(path1)
        path1 = torch.max_pool2d(path1, kernel_size=path1.size()[2:])
        path1 = path1.view(path1.size(0), -1)
        path2 = self.conv2(x)
        path2 = self.bn2(path2)
        path2 = torch.relu(path2)
        path2 = path2.view(path2.size(0), -1)
        combined = torch.cat((path1, path2), dim=1)
        combined = torch.relu(self.fc1(combined))
        combined = self.dropout(combined)
        out = self.fc2(combined)
        return out

def train_model(model: nn.Module,
                train_data: Tuple[torch.Tensor, torch.Tensor],
                val_data: Tuple[torch.Tensor, torch.Tensor],
                epochs: int = 1000,
                lr: float = 0.001) -> Tuple[nn.Module, List[float]]:
    """
    Train the model and return both the model and list of validation accuracies.
    Trains up to 1000 epochs or stops early if nearly perfect accuracy is reached.
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    val_accuracies = []

    # Adjust input shape for CNN.
    if isinstance(model, CNN):
        X_train = X_train.view(-1, 1, 3, 3)
        X_val = X_val.view(-1, 1, 3, 3)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    best_model_state = None
    best_acc = 0
    patience = 100
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_acc = (val_outputs.argmax(1) == y_val).float().mean().item()
            val_accuracies.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Val Acc = {val_acc:.4f}")

        # Stop early if nearly perfect accuracy is reached.
        if best_acc >= 0.999:
            print(f"Reached nearly perfect accuracy at epoch {epoch}")
            break

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, val_accuracies

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
