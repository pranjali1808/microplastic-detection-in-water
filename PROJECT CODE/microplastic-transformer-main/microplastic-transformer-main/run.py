import matplotlib.pyplot as plt
from pyformer import Transformer
import fileio as io
import os
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F

# Training data and labels
TRAIN_X, TRAIN_Y = 'data/no_fp/train/train-x.csv', 'data/no_fp/train/train-y.csv'
VAL_X, VAL_Y = 'data/no_fp/val/val-x.csv', 'data/no_fp/val/val-y.csv'
TEST_X, TEST_Y = 'data/no_fp/test/test-x.csv', 'data/no_fp/test/test-y.csv'
STD_X, STD_Y = 'data/no_fp/std/std-x.csv', 'data/no_fp/std/std-y.csv'

# Training Parameters
EPOCH = 200
BATCH_SIZE = 32
LR = 0.0001
# Transformer Parameters
q = 8  # Query size
v = 8  # Value size
h = 4  # Number of heads
N = 2  # Number of encoder and decoder to stack
dropout = 0.2  # Dropout rate
pe = False  # Positional encoding
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["WANDB_MODE"] = "offline"

# List to store loss and accuracy for each epoch
training_loss_list = []
training_accuracy_list = []

def init():
    train_dataloader, shape = io.get_dataloader(TRAIN_X, TRAIN_Y, BATCH_SIZE)
    val_dataloader, _ = io.get_dataloader(VAL_X, VAL_Y, BATCH_SIZE)
    test_dataloader = io.get_test_dataloader(TEST_X, TEST_Y, BATCH_SIZE)
    std_dataloader = io.get_test_dataloader(STD_X, STD_Y, BATCH_SIZE, csv=True)

    net = Transformer(d_input=shape,
                      d_channel=1,
                      d_model=shape,
                      d_output=2,
                      q=q,
                      v=v,
                      h=h,
                      N=N,
                      dropout=dropout, pe=pe).to(DEVICE)

    # Training loop
    lossy = nn.CrossEntropyLoss(reduction='mean').to(DEVICE)
    optimizer = optim.Adagrad(net.parameters(), lr=LR)

    for epoch in range(EPOCH):
        lossval = 0
        correct = 0
        total = 0
        net.train()
        for batch_idx, (x, batch_y) in enumerate(train_dataloader):
            x, batch_y = x.to(DEVICE), batch_y.long().to(DEVICE)
            optimizer.zero_grad()
            out = net(x.to(DEVICE))
            loss = lossy(out, batch_y)
            loss.backward() 
            optimizer.step()
            lossval += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(out, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        # Calculate average loss and accuracy for this epoch
        loss_avg = lossval / len(train_dataloader)
        accuracy = 100 * correct / total

        training_loss_list.append(loss_avg)
        training_accuracy_list.append(accuracy)

        print(f'Epoch: {epoch+1} | Loss: {loss_avg:.4f} | Accuracy: {accuracy:.2f}%\n---------------------------------------')

        test_network(net, val_dataloader, lossy, "val_set")
        test_network(net, std_dataloader, lossy, "std_set")

    # Test network on final test set after training is complete
    test_network(net, test_dataloader, lossy, "test_set")
    test_network(net, std_dataloader, lossy, "std_set")

    # Plot the training loss and accuracy
    plot_training_metrics()


def test_network(net, dataloader_test, lossy=None, flag='test_set'):
    correct, total, lossval = 0.0, 0.0, 0.0

    with torch.no_grad():
        for x_test, y_test in dataloader_test:
            enc_inputs, dec_inputs = x_test.to(DEVICE), y_test.long().to(DEVICE)
            test_outputs = net(enc_inputs)
            loss = lossy(test_outputs, dec_inputs)
            lossval += loss.item()
            _, predicted = torch.max(test_outputs.data, dim=1)

            total += dec_inputs.size(0)
            correct += (predicted.float() == dec_inputs.float()).sum().item()

    if flag == "val_set":
        print(f"Validation loss: {lossval/len(dataloader_test):.4f}, Validation acc: {100 * correct / total:.2f}%")
    elif flag == "std_set":
        print(f"Standards loss: {lossval/len(dataloader_test):.4f}, Standards acc: {100 * correct / total:.2f}%")
    else:
        print(f"Test loss: {lossval/len(dataloader_test):.4f}, Test acc: {100 * correct / total:.2f}%")


def plot_training_metrics():
    epochs = range(1, EPOCH + 1)

    # Plotting the training loss
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_loss_list, 'r-', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracy_list, 'b-', label='Training Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    init()
