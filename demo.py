import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from tqdm import tqdm
from utils.transform import Transform
from utils.model import PutNet

def main():
    """Train the model and save the checkpoint"""

    # Create model
    device = "cuda:0"
    model = PutNet().to(device)

    # Load weights
    model.load_state_dict(torch.load("models/model_2900000_train_980000_epochs_600_neurons_range_scaler_epoch_98000.pt"))

    # Load dataset
    train_df = pd.read_csv("dataset/training_data.csv")
    valid_df = pd.read_csv("dataset/validation_data.csv")

    # Init transformer
    transform = Transform()

    # Set up training
    x = torch.Tensor(transform.transform_x(train_df[["S", "K", "T", "r", "sigma"]].to_numpy()))
    y = torch.Tensor(transform.transform_y(train_df[["value"]].to_numpy()))
    training_data = torch.concat((x, y), dim=-1).to(device)

    x = torch.Tensor(transform.transform_x(valid_df[["S", "K", "T", "r", "sigma"]].to_numpy()))
    y = torch.Tensor(valid_df[["value"]].to_numpy())
    validation_data = torch.concat((x, y), dim=-1).to(device)

    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(model.parameters())

    n_epochs = 100000

    min_valid_max_error = float("inf")

    for i in tqdm(range(n_epochs)):
        training_loss = 0.0
        training_max_error = 0.0

        x = training_data[:, :5]
        y = training_data[:, -1]

        y_hat = model(x)

        # Calculate training loss
        loss = criterion(y_hat, y)
        training_loss += loss.item()
        training_max_error += torch.max(torch.abs(y - y_hat))

        # Take a step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1000 == 0:
            # Check validation loss
            model.eval()
            with torch.no_grad():
                x = validation_data[:, :5]
                y = validation_data[:, -1].detach().cpu()
                y_hat = torch.Tensor(transform.inverse_transform_y(model(x).detach().cpu().numpy()))
                validation_loss = criterion(y_hat, y).item()
                validation_max = torch.max(torch.abs(y - y_hat)).item()

            # Save the best model
            if validation_max < min_valid_max_error:
                min_valid_max_error = validation_max
                print("Saving model")
                torch.save(model.state_dict(), f"models/model_2900000_train_100000_epochs_600_neurons_range_scaler_epoch_{i+1}.pt")

            print(
                f"Epoch: {i + 1} | Training Loss: {training_loss:.4f} | Training Max Error: {training_max_error:.4f} ",
                f"| Validation Loss: {validation_loss:.4f} | Validation Max Error {validation_max:.4f}"
            )

            model.train()
    print("Saving model")
    torch.save(model.state_dict(), f"models/model_2900000_train_100000_epochs_600_neurons_range_scaler_epoch_{i+1}.pt")


if __name__ == "__main__":
    main()
