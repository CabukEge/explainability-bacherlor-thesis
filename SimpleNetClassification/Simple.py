import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Seed setzen für Reproduzierbarkeit
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seedList = [42, 18215, 14564, 74079, 24555, 60045, 3, 58064034, 25190, 34988]

# Funktion zum Laden der Bilder und Konvertieren in Tensoren
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in glob.glob(os.path.join(folder, '*.png')):  # oder *.jpg je nach Bildformat
        img = Image.open(filename).convert('L')  # In Graustufen umwandeln
        img = img.resize((3, 3))  # Falls nicht schon 3x3 Pixel
        img_array = np.array(img)
        images.append(img_array)
        labels.append(label)
    return images, labels

# Bilder und Labels laden
images_class0, labels_class0 = load_images_from_folder('class0', 0)
images_class1, labels_class1 = load_images_from_folder('class1', 1)

# Daten und Labels zusammenfügen
X = np.array(images_class0 + images_class1)
y = np.array(labels_class0 + labels_class1)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Einfache Datenmenge anzeigen
print("Datenmenge:")
print(X)
print(y)

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

# Liste zum Speichern aller Trainingsverluste
all_train_losses = []

for i in seedList:
    set_seed(i)
    
    # Modell initialisieren
    model = SimpleNet()

    # Verlustfunktion und Optimierer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training des Modells
    num_epochs = 500
    train_losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if (epoch+1) % 100 == 0:
            print(f'For Seed {i}: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    all_train_losses.append(train_losses)

    # Teste das Modell
    with torch.no_grad():
        predicted = model(X).argmax(dim=1)
        accuracy = (predicted == y).float().mean()
        print(f'For Seed {i}: Accuracy: {accuracy:.4f}')

        print(f'For Seed {i}: Predicted labels:', predicted.numpy())
        print(f'For Seed {i}: True labels:', y.numpy())

# Plotten des Trainingsverlusts für alle Seeds
plt.figure(figsize=(10, 5))
for idx, train_losses in enumerate(all_train_losses):
    plt.plot(train_losses, label=f'Seed {seedList[idx]}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs for Different Seeds')
plt.legend()
plt.show()
