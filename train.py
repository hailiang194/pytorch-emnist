import torch
from torch.utils.data import DataLoader
from model import NeuralNetwork
from dataset import EMNISTDataset 

BATCH_SIZE = 128
EPOCH = 20

def setup_dataloader(train_dataset, test_dataset, batch_size=BATCH_SIZE):
    return DataLoader(train_dataset, batch_size=batch_size), DataLoader(test_dataset, batch_size=batch_size)

def train(dataloader, model, loss_fn, optimizer, device="cpu"):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print("loss: {} [{}/{}]".format(loss, current, size))   

def test(dataloader, model, device="cpu"):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print("Test Error: \n Accuracy: {}, Avg loss: {}\n".format(100 * correct, test_loss))

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    
    model = NeuralNetwork()

    train_dataset = EMNISTDataset('./train.csv')
    test_dataset = EMNISTDataset('./test.csv')
    
    train_dataloader, test_dataloader = setup_dataloader(train_dataset, test_dataset)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), weight_decay=0.0)

    for epoch in range(EPOCH):
        print("Epoch {}\n--------------------------------------------".format(epoch + 1))
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model)
