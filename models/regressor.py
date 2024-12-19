import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    f1_score,
    confusion_matrix,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Generalized MLP class
class MLP(nn.Module):
    def __init__(
        self, input_size, hidden_sizes, output_size=1, output_activation=nn.Sigmoid()
    ):
        super(MLP, self).__init__()
        layers = []
        current_size = input_size

        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())  # Activation function
            current_size = hidden_size

        # Output layer
        layers.append(nn.Linear(current_size, output_size))
        layers.append(output_activation)  # Specified output activation

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Use nn.Flatten to automatically handle any input size
        x = torch.flatten(x, 1)  # Flatten all but the batch dimension
        return self.model(x)


# Training function
def train_mlp(model, nodes, targets, reg_criterion):
    # Forward pass
    outputs = model(nodes).squeeze(1) * 100.0
    reg_loss = reg_criterion(outputs, targets)
    return reg_loss


# Evaluation function for regression metrics
def evaluate_mlp(targets_list, predictions_list):
    targets = np.array(targets_list)
    predictions = np.array(predictions_list)

    # Calculate MAE, MSE, R² and F1-score (thresholding required)
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    # Convert to binary for F1 score
    threshold = 40  # Can be adjusted based on task
    binary_targets = (targets >= threshold).astype(int)
    binary_predictions = (predictions >= threshold).astype(int)

    f_score = f1_score(binary_targets, binary_predictions, average="weighted")

    return mae, mse, r2, f_score


# Mapping continuous values to classes
def map_to_classes(values):
    conditions = [
        values <= 30,
        (values > 30) & (values <= 40),
        (values > 40) & (values <= 55),
        values > 55,
    ]
    return np.select(conditions, [0, 1, 2, 3], default=-1)


# Class evaluation with confusion matrix and other metrics
def class_eval(ground_truth, pred):
    pred_classes = map_to_classes(np.array(pred))
    gt_classes = map_to_classes(np.array(ground_truth))

    # Confusion Matrix
    cm = confusion_matrix(gt_classes, pred_classes)
    print("Confusion Matrix:\n", cm)

    # Compute other metrics
    mse = mean_squared_error(ground_truth, pred)
    mae = mean_absolute_error(ground_truth, pred)
    r2 = r2_score(ground_truth, pred)
    f1 = f1_score(gt_classes, pred_classes, average="weighted")

    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    print(f"F1 Score: {f1}")

    # Plotting confusion matrix
    plot_confusion_matrix(cm)


def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["(0−30)", "(30-40)", "(41-55)", "(56-100)"],
        yticklabels=["(0−30)", "(30-40)", "(41-55)", "(56-100)"],
    )
    plt.xlabel("Predicted Classes")
    plt.ylabel("Ground Truth Classes")
    plt.title("Confusion Matrix")

    output_path = "confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Confusion matrix saved as {output_path}")
