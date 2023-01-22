import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class PutNet(nn.Module):
    """
    Example of a Neural Network that could be trained price a put option.
    TODO: modify me!
    """

    def __init__(self) -> None:
        super(PutNet, self).__init__()

        num_layers = 3
        layer_dim = 40
        self.layers = nn.ModuleList([nn.Linear(5, layer_dim)])
        for _ in range(num_layers):
            self.layers.append(nn.Linear(layer_dim, layer_dim))
        self.out = nn.Linear(layer_dim, 1)
        self.d = nn.Dropout(p=0.2)

    def forward(self, x):
        for layer in self.layers:
            x = self.d(F.relu(layer(x)))
        x = self.out(x)
        x = torch.squeeze(x, -1)
        return x


def main():
    """Train the model and save the checkpoint"""

    # Create model
    device = "cuda:0"
    model = PutNet().to(device)

    # Load dataset
    train_df = pd.read_csv("dataset/training_data.csv")
    valid_df = pd.read_csv("dataset/validation_data.csv")

    # Set up training
    x = torch.Tensor(train_df[["S", "K", "T", "r", "sigma"]].to_numpy())
    y = torch.Tensor(train_df[["value"]].to_numpy())
    training_data = torch.concat((x, y), dim=-1)

    x = torch.Tensor(valid_df[["S", "K", "T", "r", "sigma"]].to_numpy())
    y = torch.Tensor(valid_df[["value"]].to_numpy())
    validation_data = torch.concat((x, y), dim=-1)

    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(model.parameters())

    n_epochs = 1000
    batch_size = 2**18

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True
    )

    valid_dataloader = DataLoader(
        validation_data,
        batch_size=validation_data.shape[0],
        num_workers=8,
        pin_memory=True
    )

    for i in tqdm(range(n_epochs)):
        training_loss = 0.0
        training_max_error = 0.0
        num_batches = 0
        for batch in train_dataloader:
            batch = batch.to(device)
            x_batch = batch[:, :5]
            y_batch = batch[:, -1]

            # TODO: Modify to account for dataset size
            y_hat_batch = model(x_batch)

            # Calculate training loss
            loss = criterion(y_hat_batch, y_batch)
            training_loss += loss.item()
            training_max_error += torch.max(torch.abs(y_batch - y_hat_batch))
            num_batches += 1

            # Take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        training_loss /= num_batches
        training_max_error /= num_batches

        if (i+1) % 10 == 0:
            # Check validation loss
            model.eval()
            with torch.no_grad():
                validation_batch = next(iter(valid_dataloader)).to(device)
                x_valid = validation_batch[:, :5]
                y_valid = validation_batch[:, -1]
                y_valid_hat = model(x_valid)
                validation_loss = criterion(y_valid_hat, y_valid)
                validation_max = torch.max(torch.abs(y_valid - y_valid_hat))
            print(
                f"Epoch: {i + 1} | Training Loss: {training_loss:.4f} | Training Max Error: {training_max_error:.4f} ",
                f"| Validation Loss: {validation_loss:.4f} | Validation Max Error {validation_max:.4f}"
            )
            model.train()
    torch.save(model.state_dict(), "models/model.pt")


if __name__ == "__main__":
    main()
