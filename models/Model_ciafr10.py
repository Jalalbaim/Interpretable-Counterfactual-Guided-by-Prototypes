"""
Implementation of Classification Model used in CGP
@author: BAIM M.Jalal

"""

import numpy as np
from datetime import datetime
import os
import pickle
import random
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

from sklearn.metrics import f1_score, confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter

class CIFAR10Dataset(Dataset):
    def __init__(self, data_folder, train=True, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        if train:
            for i in range(1, 6):
                batch_file = os.path.join(data_folder, f"data_batch_{i}")
                with open(batch_file, 'rb') as f:
                    batch = pickle.load(f, encoding='latin1')
                self.data.append(batch['data'])
                self.labels.extend(batch['labels'])
            self.data = np.concatenate(self.data, axis=0)
        else:
            test_file = os.path.join(data_folder, "test_batch")
            with open(test_file, 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
            self.data = batch['data']
            self.labels = batch['labels']
        self.data = self.data.reshape(-1, 3, 32, 32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform:
            img = self.transform(img)
        return img, label

def get_model(device, name="resnet18"):
    
    if name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)

    if name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)

    return model.to(device)

def train(model, train_loader, val_loader, criterion, optimizer, device, writer=None, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] Batch {batch_idx+1}/{len(train_loader)} - Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

            #print(f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.3f}")

            if writer is not None:
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

        # ---- Validation phase ----
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"--> End of epoch {epoch+1} - Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_acc:.2f}%, Validation F1-score: {val_f1:.3f}")
        
        if writer is not None:
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Accuracy/validation', val_acc, epoch)
            writer.add_scalar('F1/validation', val_f1, epoch)

    print("Training completed.")


def evaluate(model, data_loader, criterion, device, phase="Test"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0

    print(f"{phase} - Average Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%, F1-score: {f1:.3f}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
    print(f"Classification Report:\n{classification_report(all_labels, all_preds)}")
    print(f"Confusion Matrix ({phase}):\n{cm}\n")

    return avg_loss, accuracy, f1, cm

def visualize_prediction(model, data_loader, device, class_names=None):
    model.eval()
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)

    idx = random.randint(0, images.size(0) - 1)
    img = images[idx].cpu()
    img = img.permute(1, 2, 0).numpy()
    true_label = labels[idx].item()
    pred_label = predicted[idx].item()

    plt.imshow(img)
    title = f"True: {true_label}, Predicted: {pred_label}"

    if class_names:
        title = f"True: {class_names[true_label]}, Predicted: {class_names[pred_label]}"

    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    date_str = datetime.now().strftime("%Y-%m-%d")
    log_name = f'./Counterfactual implementation/logs/{date_str}_Training_resnet18_cifar10'
    os.makedirs(log_name, exist_ok=True)
    print("Log directory:", log_name)
    writer = SummaryWriter(log_dir=log_name)

    data_folder = "./Counterfactual implementation/cifar-10-python/cifar-10-batches-py"
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Folder {data_folder} does not exist. Please check the path.")
    
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    full_train_dataset = CIFAR10Dataset(data_folder=data_folder, train=True, transform=transform_train)

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    test_dataset = CIFAR10Dataset(data_folder=data_folder, train=False, transform=transform_test)

    print(f"Total number of images: {len(full_train_dataset)}")
    print(f"-> Training: {len(train_dataset)}")
    print(f"-> Validation: {len(val_dataset)}")
    print(f"-> Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    model = get_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Starting Training...")
    train(model, train_loader, val_loader, criterion, optimizer, device, writer=writer, num_epochs=25)
    writer.close()

    print("Saving model weights...")
    weights_dir = "./Counterfactual implementation/weights"
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "cifar10_resnet18.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to: {weights_path}")

    model_weights = "./Counterfactual implementation/weights/cifar10_swinv2b.pth"

    model.load_state_dict(torch.load(model_weights))

    print("Evaluation on the test set:")
    evaluate(model, test_loader, criterion, device, phase="Test")
    
    class_names = ['plane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    visualize_prediction(model, test_loader, device, class_names)


if __name__ == "__main__":
    main()
