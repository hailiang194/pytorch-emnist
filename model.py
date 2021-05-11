from torch import nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Reshape from flatten to 28 x 28
        self.__reshape = lambda x: x.reshape((-1, 1, 28, 28,))

        #Training model
        self.__model = nn.Sequential(
            nn.Conv2d(1, 32, (5, 5)), # user conv 2d with kernel to create output with size is 24 x 24 x 32
            nn.ReLU(),
            nn.MaxPool2d((2, 2)), # max pooling the input layer to create output with size is 12 x 12 x 32
            nn.Flatten(), # flatten to user regular neural network to create output with size 4608 x 1
            nn.Linear(4608, 512), # create output with size 512 x 1
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 62), # create output with size 62 x 1
            nn.Softmax()
        )

    def forward(self, x):
        x = self.__reshape(x)
        y = self.__model(x)
        return y
