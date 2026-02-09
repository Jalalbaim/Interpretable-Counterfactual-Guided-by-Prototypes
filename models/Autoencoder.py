"""
Implementation of AutoEncoder used in CGP
@author: BAIM M.Jalal
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_model(model, train_loader, num_epochs, device, lr=0.001):

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return train_losses

def visualize(model, dataset, device, index=0):
    model.eval()
    img, label = dataset[index]
    img_flat = img.unsqueeze(0).to(device)
    with torch.no_grad():
        reconstruction = model(img_flat)

    original_np = img_flat.squeeze(0).cpu().numpy()
    reconstructed_np = reconstruction.squeeze(0).cpu().numpy()

    # Plot side by side
    fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
    axes[0].imshow(original_np[0], cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(reconstructed_np[0], cmap='gray')
    axes[1].set_title("Reconstructed")
    axes[1].axis('off')

    plt.suptitle(f"Sample index: {index}")
    plt.show()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Hyperparams
    num_epochs = 25
    batch_size = 64
    learning_rate = 1e-5

    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = AutoEncoder()

    # Train
    print("Starting training...")
    train_losses = train_model(model, train_loader, num_epochs=num_epochs, lr=learning_rate, device=device)

    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, 'o--')
    plt.title("Training Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    torch.save(model.state_dict(), "./weights/autoencoder_mnist.pth")
    print("Model saved to autoencoder_mnist.pth")

    visualize(model, train_dataset, device=device, index=0)

if __name__ == "__main__":
    main()