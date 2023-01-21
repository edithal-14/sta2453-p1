import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


class PutNet(nn.Module):
    """
    Example of a Neural Network that could be trained price a put option.
    TODO: modify me!
    """

    def __init__(self) -> None:
        super(PutNet, self).__init__()

        self.l1 = nn.Linear(5, 20)
        self.d1 = nn.Dropout(p=0.2)
        self.l2 = nn.Linear(20, 20)
        self.d2 = nn.Dropout(p=0.2)
        self.l3 = nn.Linear(20, 20)
        self.d3 = nn.Dropout(p=0.2)
        self.out = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.d1(x)
        x = F.relu(self.l2(x))
        x = self.d2(x)
        x = F.relu(self.l3(x))
        x = self.d3(x)
        x = self.out(x)
        x = torch.squeeze(x, -1)
        return x


def main():
    """Train the model and save the checkpoint"""

    # Create model
    device = "cuda:0"
    model = PutNet().to(device)

    # Load dataset
    df = pd.read_csv("training_data.csv")

    # Set up training
    x = torch.Tensor(df[["S", "K", "T", "r", "sigma"]].to_numpy())
    y = torch.Tensor(df[["value"]].to_numpy())
    training_data = torch.concat((x, y), dim=-1).to(device)

    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(model.parameters())

    n_epochs = 10000
    batch_size = 64

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    for i in range(n_epochs):

        batch = next(iter(train_dataloader))
        x = batch[:, :5]
        y = batch[:, -1]

        # TODO: Modify to account for dataset size
        y_hat = model(x)

        # Calculate training loss
        training_loss = criterion(y_hat, y)

        # Take a step
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

        # Check validation loss
        with torch.no_grad():
            # TODO: use a proper validation set
            validation_loss = criterion(model(x), y)
            validation_max = torch.max(torch.abs(y - model(x)))
        if i % 1000 == 0:
            print(f"Epoch: {i} | Training Loss: {training_loss:.4f} | Validation Loss: {validation_loss:.4f} | Max Error {validation_max:.4f} ")
    print(f"Epoch: {i} | Training Loss: {training_loss:.4f} | Validation Loss: {validation_loss:.4f} | Max Error {validation_max:.4f} ")
    torch.save(model.state_dict(), "simple-model.pt")


if __name__ == "__main__":
    main()
