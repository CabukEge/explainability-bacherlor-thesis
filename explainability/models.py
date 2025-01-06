import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple


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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x.view(-1, 9))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)  # Single 3x3 filter
        self.fc1 = nn.Linear(8, 2)  # Fully connected layer (2 outputs: true/false)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x


def train_model(model: nn.Module,
                train_data: Tuple[torch.Tensor, torch.Tensor],
                val_data: Tuple[torch.Tensor, torch.Tensor],
                epochs: int = 1000,
                lr: float = 0.01) -> nn.Module:
    X_train, y_train = train_data
    X_val, y_val = val_data

    # Adjust input shape for CNN
    if isinstance(model, CNN):
        X_train = X_train.view(-1, 1, 3, 3)
        X_val = X_val.view(-1, 1, 3, 3)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    best_model = None
    best_acc = 0
    patience = 100
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_acc = (val_outputs.argmax(1) == y_val).float().mean()

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 20 == 0:
                print(f'Epoch {epoch}: Val Acc = {val_acc:.4f}')

            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    model.load_state_dict(best_model)
    return model


def train_tree(X_train: torch.Tensor, y_train: torch.Tensor) -> DecisionTreeClassifier:
    """Train a Decision Tree Classifier for DNF tasks."""
    clf = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=6,  # Increased depth for more complex DNFs
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    clf.fit(X_train.view(-1, 9), y_train)
    return clf
