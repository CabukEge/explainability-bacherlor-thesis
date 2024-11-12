from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Constants
SEED_LIST = [42, 18215, 14564, 74079, 24555, 60045, 3, 58064034, 25190, 34988]
ST_NUM_EPOCHS = 500

def decision_tree(X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
    """
    Train a decision tree classifier.
    
    Args:
        X: Input features
        y: Target labels
        
    Returns:
        DecisionTreeClassifier: Trained model
    """
    # Initialize and train the decision tree
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    
    # Make predictions
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f'Decision Tree Accuracy: {accuracy:.4f}')
    
    return clf  # Return the trained model

# Einfaches Netz
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3 * 3, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = x.view(-1, 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Convolutional Neural Network
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 3 * 3, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        # Flatten and fully connected layers
        x = x.view(-1, 32 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def train_fully_connected(X: torch.Tensor, 
                         y: torch.Tensor, 
                         seed_list: list, 
                         num_epochs: int) -> tuple:
    """
    Train multiple FCN models with different seeds.
    
    Args:
        X: Input data
        y: Target labels
        seed_list: List of random seeds
        num_epochs: Number of training epochs
        
    Returns:
        tuple: (losses, models)
    """
    all_train_losses = []
    final_models = []
    
    for seed in seed_list:
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Initialize model and optimizer
        model = SimpleNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Training loop
        train_losses = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            if (epoch+1) % 100 == 0:
                print(f'FCN Seed {seed}: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # Evaluate final model
        with torch.no_grad():
            predicted = model(X).argmax(dim=1)
            accuracy = (predicted == y).float().mean()
            print(f'FCN Seed {seed}: Accuracy: {accuracy:.4f}')
        
        all_train_losses.append(train_losses)
        final_models.append(model)
    
    return all_train_losses, final_models

def train_cnn(X_train: torch.Tensor, 
              y_train: torch.Tensor,
              X_val: torch.Tensor = None,
              y_val: torch.Tensor = None,
              seed_list: list = SEED_LIST,
              num_epochs: int = ST_NUM_EPOCHS) -> tuple:
    """
    Train multiple CNN models with different seeds and improved training process.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data (optional)
        y_val: Validation labels (optional)
        seed_list: List of random seeds
        num_epochs: Number of training epochs
        
    Returns:
        tuple: (losses, models)
    """
    all_train_losses = []
    all_val_losses = []
    final_models = []
    
    # Add channel dimension if not present
    X_train = X_train.unsqueeze(1) if len(X_train.shape) == 3 else X_train
    if X_val is not None:
        X_val = X_val.unsqueeze(1) if len(X_val.shape) == 3 else X_val
    
    for seed in seed_list:
        print(f"\nTraining CNN with seed {seed}")
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Initialize model and optimizer
        model = ImprovedCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=20, 
                                                       verbose=True)
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        early_stopping_counter = 0
        early_stopping_patience = 50
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            train_loss = criterion(outputs, y_train)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                if X_val is not None:
                    val_outputs = model(X_val)
                    val_loss = criterion(val_outputs, y_val)
                    val_losses.append(val_loss.item())
                    
                    # Learning rate scheduling
                    scheduler.step(val_loss)
                    
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict().copy()
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                
                # Print progress
                if (epoch + 1) % 50 == 0:
                    train_pred = outputs.argmax(dim=1)
                    train_acc = (train_pred == y_train).float().mean()
                    print(f'Epoch [{epoch+1}/{num_epochs}]')
                    print(f'Train Loss: {train_loss.item():.4f}, Accuracy: {train_acc:.4f}')
                    if X_val is not None:
                        val_pred = val_outputs.argmax(dim=1)
                        val_acc = (val_pred == y_val).float().mean()
                        print(f'Val Loss: {val_loss.item():.4f}, Accuracy: {val_acc:.4f}')
            
            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model if we had validation data
        if X_val is not None and best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train)
            train_pred = train_outputs.argmax(dim=1)
            train_acc = (train_pred == y_train).float().mean()
            print(f'\nFinal Training Accuracy: {train_acc:.4f}')
            
            if X_val is not None:
                val_outputs = model(X_val)
                val_pred = val_outputs.argmax(dim=1)
                val_acc = (val_pred == y_val).float().mean()
                print(f'Final Validation Accuracy: {val_acc:.4f}')
        
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses if X_val is not None else [])
        final_models.append(model)
    
    return all_train_losses, all_val_losses, final_models