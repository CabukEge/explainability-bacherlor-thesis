import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from .models import SimpleNet, ImprovedCNN

# Constants
SEED_LIST = [42, 18215, 14564, 74079, 24555, 60045, 3, 58064034, 25190, 34988]
ST_NUM_EPOCHS = 500

def evaluate_consecutive_ones(model, X_test, y_test):
    """Detailed evaluation of consecutive ones task"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).argmax(dim=1)
        accuracy = (predictions == y_test).float().mean()
        
        # Find failure cases
        errors = (predictions != y_test).nonzero().squeeze()
        if len(errors) > 0:
            print("\nFailure cases:")
            for idx in errors:
                input_vec = X_test[idx].view(-1).numpy()
                pred = predictions[idx].item()
                true = y_test[idx].item()
                has_consecutive = any(all(input_vec[i:i+3] == 1) for i in range(len(input_vec)-2))
                print(f"Input: {input_vec}")
                print(f"Has 3 consecutive ones: {has_consecutive}")
                print(f"Predicted: {pred}, True: {true}\n")
                
        return accuracy

def decision_tree(X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
    """Train a decision tree classifier."""
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f'Decision Tree Accuracy: {accuracy:.4f}')
    
    return clf

def train_fully_connected(X: torch.Tensor, 
                         y: torch.Tensor, 
                         seed_list: list = SEED_LIST, 
                         num_epochs: int = ST_NUM_EPOCHS,
                         learning_rate: float = 0.01) -> tuple:
    """Train multiple FCN models with different seeds."""
    all_train_losses = []
    final_models = []
    
    for seed in seed_list:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = SimpleNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        best_loss = float('inf')
        best_model = None
        patience = 20
        patience_counter = 0
        
        train_losses = []
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            scheduler.step(loss)
            
            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
                
            if (epoch+1) % 100 == 0:
                print(f'FCN Seed {seed}: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        if best_model is not None:
            model.load_state_dict(best_model)
            
        model.eval()
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
    """Train multiple CNN models with different seeds."""
    all_train_losses = []
    all_val_losses = []
    final_models = []
    
    X_train = X_train.unsqueeze(1) if len(X_train.shape) == 3 else X_train
    if X_val is not None:
        X_val = X_val.unsqueeze(1) if len(X_val.shape) == 3 else X_val
    
    for seed in seed_list:
        print(f"\nTraining CNN with seed {seed}")
        torch.manual_seed(seed)
        
        model = ImprovedCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        
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
                    scheduler.step(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict().copy()
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                
                if (epoch + 1) % 50 == 0:
                    train_pred = outputs.argmax(dim=1)
                    train_acc = (train_pred == y_train).float().mean()
                    print(f'Epoch [{epoch+1}/{num_epochs}]')
                    print(f'Train Loss: {train_loss.item():.4f}, Accuracy: {train_acc:.4f}')
                    if X_val is not None:
                        val_pred = val_outputs.argmax(dim=1)
                        val_acc = (val_pred == y_val).float().mean()
                        print(f'Val Loss: {val_loss.item():.4f}, Accuracy: {val_acc:.4f}')
            
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        if X_val is not None and best_model_state is not None:
            model.load_state_dict(best_model_state)
        
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