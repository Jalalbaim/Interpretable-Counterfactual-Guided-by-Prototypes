"""
Counterfactuals Guided By Prototypes main function
as described in the paper
@author: BAIM M.J
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.Autoencoder import AutoEncoder
from models.Model_MNIST import Model
import matplotlib.pyplot as plt
import torch
from algorithm import Counterfactuals
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def trained_model(model, autoencoder, weights_classif, weights_auto, device):
    classif = torch.load(weights_classif)
    model.load_state_dict(classif)
    model.to(device)

    autoenc = torch.load(weights_auto)
    autoencoder.load_state_dict(autoenc)
    autoencoder.to(device)

    encoder = autoencoder.encoder

    return model, autoencoder, encoder


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    print("Using device : ", device)

    #torch.cuda.empty_cache()
    #print("cache empty")

    # date and time
    time = datetime.now().strftime("%H-%M-%S")
    date_str = datetime.now().strftime("%Y-%m-%d") + f"_{time}"

    #data
    transform = transforms.Compose([
        transforms.ToTensor()])

    train_dataset = datasets.MNIST(root='./', train=True,
                                   download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./', train=False,
                                  download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    print(len(images))
    index = random.randint(0, len(images)-1)
    print(f"Selected index: {index}")
    log_name = f'./logs/{date_str}_MNIST_{index}'
    writer = SummaryWriter(log_dir=log_name)

    x_orig = images[index].unsqueeze(0).to(device)
    train_data = torch.stack([image[0] for image in train_loader.dataset], dim=0).to(device)

    # Model, Autoencoder
    model = Model()
    autoencoder = AutoEncoder()

    model_weights = "./weights/mnist_cgp_model_weights.pth"
    autoencoder_weights = "./weights/autoencoder_mnist.pth"

    model_trained, autoencoder_trained, encoder_trained = trained_model(model, autoencoder, model_weights, autoencoder_weights, device)
    print("Trained model, autoencoder, encoder loaded.")

    counterfactual = Counterfactuals(model_trained, encoder_trained, autoencoder_trained, device=device)

    print("Generating counterfactual...")
    x_counterfactual = counterfactual.algorithm_CGP(
        x_orig,
        train_data,
        c=1.0,
        beta=0.1,
        theta=200,
        cap=0.01,
        gamma=100,
        K=5,
        max_iterations=5000,
        lr=1e-2,
        device=device,
        writer= writer
    )

    with torch.no_grad():
        pred_cf = model_trained(x_counterfactual).argmax(dim=1).item()
        pred_org = model_trained(x_orig).argmax(dim=1).item()

    # Plot side by side
    fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
    axes[0].imshow(x_orig.detach().cpu().numpy()[0][0], cmap='gray')
    axes[0].set_title(f'Original - Class {pred_org}')
    axes[0].axis('off')

    axes[1].imshow(x_counterfactual.detach().cpu().numpy()[0][0], cmap='gray')
    axes[1].set_title(f'Counterfactual - Class {pred_cf}')
    axes[1].axis('off')

    plt.show()

    # save the figure
    plt.savefig(f'./outputs/{date_str}_MNIST_NearestProto_{index}.png')
    plt.close()

    writer.close()

if __name__ == "__main__":
    main()