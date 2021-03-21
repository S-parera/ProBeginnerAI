import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
import math
from tqdm import tqdm
import time


def main():
    # Hyperparameters
    n_epochs = 10
    batch_size = 128
    learning_rate = 0.001

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms for the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./CNN/torch/data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                                            shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./CNN/torch/data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, pin_memory=True,
                                            shuffle=False, num_workers=4)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # functions to show an image
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap=plt.cm.binary)
        plt.show()


    # get one batch of images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images, normalize=True, nrow=int(np.sqrt(batch_size))))

    def build_model():
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(32, 64, 3)
                self.conv3 = nn.Conv2d(64, 64, 3)
                self.fc1 = nn.Linear(64*4*4, 64)
                self.fc2 = nn.Linear(64, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = F.relu(self.conv3(x))
                x = x.view(-1, 64*4*4)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        return Net()


    net = build_model().to(device)

    print(summary(net, (3,32,32)))

    # Define model loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    n_batches = math.ceil(len(trainset)/batch_size)

    start = time.perf_counter()
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print(f"\nEpoch: {epoch}/{n_epochs}")

        running_loss = 0.0
        for data in tqdm(trainloader, ascii=True, ncols=75):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # Send to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print(f"Loss: {running_loss/n_batches:.2f}")


    finish = time.perf_counter()

    print(f"Finished training in: {finish-start:.2f}")

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:

            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100*correct/total:.2f} %')


if __name__=='__main__':
    main()