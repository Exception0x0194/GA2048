import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 128, dtype=float)
        self.fc2 = nn.Linear(128, 128, dtype=float)
        self.fc3 = nn.Linear(128, 4, dtype=float)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=0)


def preprocess(board):
    processed_board = torch.log2(torch.clip(board, 1, torch.inf))
    return processed_board.flatten().to(float)


def postprocess(x):
    idx = torch.argmax(x)
    if idx == 0:
        return "up"
    elif idx == 1:
        return "down"
    elif idx == 2:
        return "left"
    return "right"


# Test
if __name__ == "__main__":
    net = Net()
    board = torch.tensor(
        [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 0], [0, 0, 0, 0]],
        dtype=float,
    )
    input_tensor = preprocess(board)
    output = net(input_tensor)
    print(postprocess(output))
