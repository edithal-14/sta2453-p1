import pandas as pd
import argparse
import sys

# Load whatever imports you need, but make sure to add them to the requirements.txt file.

# My imports
import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.model import PutNet
from utils.transform import Transform


def riskfuel_test(df: pd.DataFrame) -> float:
    """
    Riskfuel Testing Function
    by <student-name>: <member_1> <member_2> .... <member_k>

    arguments: pandas DataFrame type with the following columns.. ['S','K','T','r','sigma','value'] all are of type float32
    ouputs: mean absolute error (float)

    Once you have finished model training/developemnt you must save the model within the repo and load it in using this function.

    You are free to import any python packages you desire but you must add them to the requirements.txt file.

    This function must do the following:
        - Successfully load your own model.
        - Take in a dataframe consisting of (N x 6) float32's.
        - Take the (N x 5) columns regarding the inputs to the pricer ['S','K','T','r','sigma'] and have your model price them.
        - Return the Mean  Absolute Error of the model.

    Do not put the analytic pricer as part of your network.
    Do not do any trickery with column switching as part of your answer.

    These will be checked by hand, any gaslighting will result in automatic disqualification.

    The following example has been made available to you.
    """

    # DEFINITIONS.
    student_name = "Vignesh Edithal"  # adjust this.
    full_name = "Vignesh Edithal" # adjust this.
    student_id_name = "edithals"

    print(f"\n\n ============ Evaluating Student: {student_name} ========================= ")
    print(" Full name:")
    print("", full_name)
    print(" Student ID:")
    print("", student_id_name)
    print(" ================================================================ \n")

    # ===============   Example Code  ===============

    # My model uses PyTorch but you can use whatever package you like,
    # as long you write code to load it and effectively calculate the mean absolute aggregate error.

    # LOAD MODEL
    mm = PutNet()
    # mm.load_state_dict(torch.load("simple-model.pt"))
    mm.load_state_dict(torch.load("models/model_1000000_train_10000_epoch_192_neurons_range_scaler.pt"))
    mm.eval()  # evaluation mode

    # EVALUATE MODEL

    # Initialize transform function
    transform = Transform(use_boxcox=True)

    # Acquire inputs/outputs
    x = torch.Tensor(transform.transform_x(df[["S", "K", "T", "r", "sigma"]].to_numpy()))
    y = torch.Tensor(df[["value"]].to_numpy()).flatten()

    # Pass data through model
    y_hat = torch.Tensor(transform.inverse_transform_y(mm(x).detach().cpu().numpy()))

    # Calculate mean squared error
    result = F.mse_loss(y_hat, y)

    # Max loss 
    max_loss = torch.max(torch.abs(y - y_hat))

    # Return performance metric; must be of type float
    return result.item(), max_loss.item()


def get_parser():
    """Parses the command line for the dataframe file name"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_frame_name", type=str)
    return parser


def main(args):
    """Parses arguments and evaluates model performance"""

    # Parse arguments.
    parser = get_parser()
    args = parser.parse_args(args)

    # Load DataFrame and pass through riskfuel_test function.
    df = pd.read_csv(args.data_frame_name)
    performance_metric, max_loss = riskfuel_test(df)

    # Must pass this assertion
    assert isinstance(performance_metric, float)

    print(f" MODEL PERFORMANCE: {performance_metric} \n\n")
    print(f" MAX LOSS: {max_loss} \n\n")


if __name__ == "__main__":
    main(sys.argv[1:])
