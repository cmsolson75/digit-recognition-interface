import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from net import Net, pre_processing

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

device = "cpu"
model = Net().to(device)
model.load_state_dict(torch.load("model/model.pth", weights_only=True, map_location=torch.device(device)))

test_dataloader = DataLoader(test_data, batch_size=10)


loss_fn = nn.CrossEntropyLoss()



def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            print(X.shape, y.shape)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # break
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss {test_loss:>8f} \n")

def compare_files(np_array_path, data):
    arr = np.load(np_array_path)
    img_tensor = pre_processing(arr)
    print(img_tensor)
    print(img_tensor.mean())
    plt.imshow(img_tensor.squeeze())
    plt.show()

    X, y = data[0][0], data[0][1]
    print(X)
    print(X.mean())
    # plt.imshow(X.squeeze())
    # plt.show()


def plot_image(img_path: str):
    img = Image.open(img_path)
    plot = plt.imshow(img)
    plt.show()




if __name__ == "__main__":
    # test(test_dataloader, model, loss_fn)
    # plot_image("image_plot_after_transform.png")
    compare_files("image_array.npy", test_data)
