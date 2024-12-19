import os
import random
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import echonet
from echonet.models.graph_matching import GModuleSelfAttention
from echonet.utils.segmentation import (
    load_model,
    set_random_seed,
    load_graph_model,
    load_regressor_model,
)
from echonet.models.model import DeepLabFeatureExtractor
from echonet.models.regressor import MLP, evaluate_mlp, class_eval
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set CUDA environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint, filter mismatched parameters, and update model state."""
    checkpoint = torch.load(checkpoint_path)
    pretrained_dict = checkpoint["state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Checkpoint loaded, mismatched parameters were skipped.")


def evaluate(
    data_dir,
    output,
    model_name="deeplabv3_resnet50",
    pretrained=False,
    num_workers=1,
    device=None,
    seed=0,
):
    """Main evaluation function for segmentation."""
    set_random_seed(seed)

    # Set device configuration
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set output directory
    if output is None:
        output = os.path.join(
            "output",
            "segmentation",
            f"{model_name}_{'pretrained' if pretrained else 'random'}",
        )
    os.makedirs(output, exist_ok=True)

    # Load models
    model = load_model(device, pretrained)
    graph_model = load_graph_model(device)
    reg_model = load_regressor_model(device)

    # Load checkpoints
    load_checkpoint(
        graph_model,
        "/home/bassant/dynamic/output/segmentation/deeplabv3_resnet50_random/final_graph_top_best.pt",
    )
    load_checkpoint(
        reg_model,
        "/home/bassant/dynamic/output/segmentation/deeplabv3_resnet50_random/linear_reg_checkpoint.pt",
    )

    # Freeze graph and regressor models
    for model in [graph_model, reg_model]:
        for param in model.parameters():
            param.requires_grad = False
    graph_model.eval()
    model.eval()
    reg_model.eval()

    # Compute mean and std for normalization
    mean, std = echonet.utils.get_mean_and_std(
        echonet.datasets.Echo(root=data_dir, split="test")
    )
    kwargs = {
        "target_type": ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace", "EF"],
        "mean": mean,
        "std": std,
    }

    # Load dataset and dataloader
    dataset = {"test": echonet.datasets.Echo(root=data_dir, split="test", **kwargs)}
    dataloader = torch.utils.data.DataLoader(
        dataset["test"],
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Initialize metrics
    total_loss = 0.0
    large_inter, large_union, small_inter, small_union = 0, 0, 0, 0
    large_inter_list, large_union_list, small_inter_list, small_union_list = (
        [],
        [],
        [],
        [],
    )
    losses = {}
    EF_list, prediction_list = [], []

    # Iterate over dataloader
    with tqdm.tqdm(total=len(dataloader)) as pbar:
        for _, (large_frame, small_frame, large_trace, small_trace, EF) in dataloader:
            large_frame, large_trace = large_frame.to(device), large_trace.to(device)
            small_frame, small_trace = small_frame.to(device), small_trace.to(device)
            EF = EF.to("cpu")

            # Prediction for large and small frames
            y_large_d, features_d = model(large_frame)
            y_small_s, features_s = model(small_frame)

            # Upsample predictions
            y_large_upsampled = F.interpolate(
                y_large_d[0], size=(112, 112), mode="bilinear", align_corners=False
            )
            y_small_upsampled = F.interpolate(
                y_small_s[0], size=(112, 112), mode="bilinear", align_corners=False
            )

            # Compute binary cross entropy loss
            loss_large = F.binary_cross_entropy_with_logits(
                y_large_upsampled[:, 0, :, :], large_trace, reduction="sum"
            )
            loss_small = F.binary_cross_entropy_with_logits(
                y_small_upsampled[:, 0, :, :], small_trace, reduction="sum"
            )

            # Compute intersection and union metrics
            large_inter += np.logical_and(
                y_large_upsampled[:, 0, :, :].detach().cpu().numpy() > 0.0,
                large_trace.detach().cpu().numpy() > 0.0,
            ).sum()
            large_union += np.logical_or(
                y_large_upsampled[:, 0, :, :].detach().cpu().numpy() > 0.0,
                large_trace.detach().cpu().numpy() > 0.0,
            ).sum()
            small_inter += np.logical_and(
                y_small_upsampled[:, 0, :, :].detach().cpu().numpy() > 0.0,
                small_trace.detach().cpu().numpy() > 0.0,
            ).sum()
            small_union += np.logical_or(
                y_small_upsampled[:, 0, :, :].detach().cpu().numpy() > 0.0,
                small_trace.detach().cpu().numpy() > 0.0,
            ).sum()

            # Compute graph loss and prediction
            dice_loss = (loss_large + loss_small) / 2
            refined_nodes, G_loss = graph_model(
                features_s, y_small_s, features_d, y_large_d
            )
            refined_nodes = refined_nodes.to(device)
            prediction = reg_model(refined_nodes).squeeze(1).to("cpu")

            # Accumulate results
            EF_list.append(EF)
            prediction_list.append(prediction)
            losses.update(G_loss)
            total_loss += dice_loss.item()

            # Update progress bar
            pbar.set_postfix_str(
                f"DICE Loss: {dice_loss.item():.4f} | Mat Loss: {G_loss['mat_loss_aff']:.4f} | Node Loss: {G_loss['node_loss']:.4f}"
            )
            pbar.update()

    # Evaluate regression metrics
    mae, mse, r2, f_score = evaluate_mlp(EF_list, prediction_list)
    class_eval(EF_list, prediction_list)

    return (
        total_loss,
        G_loss["mat_loss_aff"],
        G_loss["node_loss"],
        mae,
        mse,
        r2,
        f_score,
    )


if __name__ == "__main__":
    total_loss, mat_loss, classification_loss, reg_epoch_mae, mse, r2, f_score = (
        evaluate(data_dir="path/to/data", output="path/to/output")
    )
    print(f"Total Loss: {total_loss}")
    print(f"Matrix Loss: {mat_loss}")
    print(f"Classification Loss: {classification_loss}")
    print(f"Regression Epoch MAE: {reg_epoch_mae}")
    print(f"Regression Epoch MSE: {mse:.4f}")
    print(f"Regression Epoch R2_: {r2:.4f}")
    print(f"Regression Epoch F_S: {f_score:.4f}")
