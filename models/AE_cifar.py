"""
Implementation of AE for the cifar dataset.
"""

import random
import matplotlib.pyplot as plt
from datetime import datetime
import yaml

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from models.Model_ciafr10 import CIFAR10Dataset


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    def get_latent(self, x):
        encoded = self.encoder(x)
        return encoded.view(encoded.size(0), -1)
    
    def evaluate_model(self, test_loader, device):
        self.eval()
        total_loss = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = self(images)
                loss = criterion(outputs, images)
                total_loss += loss.item()
        avg_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_loss:.4f}")
        return avg_loss
    

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device : ", device)

    date_str = datetime.now().strftime("%Y-%m-%d")
    log_name = f'./Counterfactual implementation/logs/{date_str}_AE_cifar10'
    #writer = SummaryWriter(log_dir=log_name)
    print("Log directory:", log_name)

    saved = './Counterfactual implementation/weights/AE_cifar10'

    # data
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    data_folder = "./Counterfactual implementation/cifar-10-python/cifar-10-batches-py"

    full_train_dataset = CIFAR10Dataset(data_folder=data_folder, train=True, transform=transform_train)

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    test_dataset = CIFAR10Dataset(data_folder=data_folder, train=False, transform=transform_test)

    print(f"Total number of images: {len(full_train_dataset)}")
    print(f"-> Training: {len(train_dataset)}")
    print(f"-> Validation: {len(val_dataset)}")
    print(f"-> Test: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # train 
    print("training started")
    model = Autoencoder()
    model.to(device)
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # num_epochs = 100
    # best_val_loss = float('inf')

    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0
    #     for images, _ in train_loader:
    #         images = images.to(device)

    #         optimizer.zero_grad()
    #         _, outputs = model(images)
    #         loss = criterion(outputs, images)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()

    #     # Log the loss
    #     writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)

    #     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    #     # Validation phase
    #     model.eval()
    #     val_loss = 0.0
    #     with torch.no_grad():
    #         for images, _ in val_loader:
    #             images = images.to(device)
    #             _, outputs = model(images)
    #             loss = criterion(outputs, images)
    #             val_loss += loss.item()
    #     val_loss /= len(val_loader)
    #     writer.add_scalar('Loss/val', val_loss, epoch)
    #     print(f"Validation Loss: {val_loss:.4f}")

    #     # Save the model based on best validation loss
    #     if epoch == 0 or val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         torch.save(model.state_dict(), saved)
    #         print(f"Model saved with validation loss: {best_val_loss:.4f}")

    print("Training finished")

    model_weights = torch.load(saved)
    model.load_state_dict(model_weights)

    # Test the model
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    index = random.randint(0, len(images)-1)
    print(index)

    x_orig = images[index].unsqueeze(0)
    x_orig = x_orig.to(device)
    print(x_orig.shape)

    with torch.no_grad():
        pred_org = model(x_orig)
        pred_org = pred_org.cpu().numpy()
        print("Original class:", labels[index].item())

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(x_orig.cpu().numpy().squeeze().transpose(1, 2, 0))
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_org.squeeze().transpose(1, 2, 0))
    plt.title("Reconstructed")
    plt.show()

    # Evaluate the model on test dataset
    test_loss = model.evaluate_model(test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")


    
    # Save parameters in yaml file
    params = {
        'model' :{
            'name': 'Autoencoder',
            'input_size': [3, 32, 32],
            'output_size': [3, 32, 32],
            'encoder_layers': [12, 24, 48, 96],
            'decoder_layers': [96, 48, 24, 12],
            'kernel_size': 4,
            'Latent_size': [96,2,2],
        },
        'training' :{
            'batch_size': 128,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'loss_function': 'MSELoss'
        },
        'data' :{
            'train_size': train_size,
            'val_size': val_size,
            'test_size': len(test_dataset),
            'Transform': {
                'Resize': (32, 32),
                'RandomHorizontalFlip': True,
                'ToTensor': True
            }
        },
        'Evaluation' :{
            'test_loss': test_loss,
        },
        'log' :{
            'log_name': log_name,
            'date': date_str,
            'saved_model_path': saved
        }
    }
    yaml_file = './Counterfactual implementation/Hyperparameters/AE_Cifar10.yaml'
    with open(yaml_file, 'w') as file:
        yaml.dump(params, file)
    print(f"Parameters saved in {yaml_file}")


if __name__ == '__main__':
    main()