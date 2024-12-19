"""Functions for training and running segmentation."""

import math
import os
import time
import click
import matplotlib.pyplot as plt
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm
import torch.nn.functional as F
import echonet
import torch.nn as nn
import wandb
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Restrict CUDA to specific GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,4,5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from echonet.models.graph_matching import GModuleSelfAttention
from echonet.models.model import DeepLabFeatureExtractor
from echonet.models.regressor import MLP, train_mlp


@click.command("segmentation")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option(
    "--model_name",
    type=click.Choice(
        sorted(
            name
            for name in torchvision.models.segmentation.__dict__
            if name.islower()
            and not name.startswith("__")
            and callable(torchvision.models.segmentation.__dict__[name])
        )
    ),
    default="deeplabv3_resnet50",
)
@click.option("--pretrained/--random", default=False)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=False)
@click.option("--save_video/--skip_video", default=False)
@click.option("--num_epochs", type=int, default=50)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=0)
@click.option("--lr_step_period", type=int, default=None)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=32)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(
    data_dir=None,
    output=None,
    model_name="deeplabv3_resnet50",
    pretrained=False,
    weights=None,
    run_test=False,
    save_video=False,
    num_epochs=50,
    lr=1e-5,
    weight_decay=1e-5,
    lr_step_period=None,
    num_train_patients=None,
    num_workers=6,
    batch_size=2,
    device=None,
    seed=0,
):
    """Trains/tests segmentation model.

    Args:
        data_dir (str, optional): Directory containing dataset. Defaults to
            `echonet.config.DATA_DIR`.
        output (str, optional): Directory to place outputs. Defaults to
            output/segmentation/<model_name>_<pretrained/random>/.
        model_name (str, optional): Name of segmentation model. One of ``deeplabv3_resnet50'',
            ``deeplabv3_resnet101'', ``fcn_resnet50'', or ``fcn_resnet101''
            (options are torchvision.models.segmentation.<model_name>)
            Defaults to ``deeplabv3_resnet50''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to False.
        weights (str, optional): Path to checkpoint containing weights to
            initialize model. Defaults to None.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
        save_video (bool, optional): Whether to save videos with segmentations.
            Defaults to False.
        num_epochs (int, optional): Number of epochs during training
            Defaults to 50.
        lr (float, optional): Learning rate for SGD
            Defaults to 1e-5.
        weight_decay (float, optional): Weight decay for SGD
            Defaults to 0.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            Defaults to math.inf (never decay learning rate).
        num_train_patients (int or None, optional): Number of training patients
            for ablations. Defaults to all patients.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        device (str or None, optional): Name of device to run on. Options from
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            Defaults to ``cuda'' if available, and ``cpu'' otherwise.
        batch_size (int, optional): Number of samples to load per batch
            Defaults to 20.
        seed (int, optional): Seed for random number generator. Defaults to 0.
    """
    # Set random seed for reproducibility
    set_random_seed(seed)
    learning_rate = 0.00001

    # Configure device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    # Set default output directory
    if output is None:
        output = os.path.join(
            "output",
            "segmentation",
            "{}_{}".format(model_name, "pretrained" if pretrained else "random"),
        )
    os.makedirs(output, exist_ok=True)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)
    model = load_model(device, pretrained)

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint["state_dict"])

    graph_model = load_graph_model(device)
    # Set up optimizer
    optim = torch.optim.SGD(
        graph_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    reg_model = load_regressor_model(device)
    reg_criterion = nn.MSELoss()
    reg_optimizer = torch.optim.Adam(reg_model.parameters(), lr=learning_rate)

    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)
    reg_scheduler = torch.optim.lr_scheduler.StepLR(reg_optimizer, lr_step_period)

    wandb.init(
        project="Grph_train",  # Consolidated project name
        config={
            "lr": lr,
            "batch_size": batch_size,
            "num_epoch": num_epochs,
        },
    )

    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(
        echonet.datasets.Echo(root=data_dir, split="train")
    )
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace", "EF"]
    kwargs = {"target_type": tasks, "mean": mean, "std": std}

    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(
            len(dataset["train"]), num_train_patients, replace=False
        )
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:

        bestLoss = float("inf")
        best_re_loss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint["state_dict"])
            optim.load_state_dict(checkpoint["opt_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_dict"])
            epoch_resume = checkpoint["epoch"] + 1
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        train_losses = []
        val_losses = []

        losses = {}
        reg_epoch_loss = 0.0
        reg_train_losses = []
        reg_val_losses = []
        epoch_resume = 0
        for epoch in range(epoch_resume, num_epochs):
            epoch_metrics = {}  # Store metrics per epoch
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ["train", "val"]:
                start_time = time.time()
                phase_losses = []
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                if phase == "train":
                    # graph_model.train()
                    model.eval()
                    graph_model.eval()
                    reg_model.train()
                else:
                    graph_model.eval()
                    reg_model.eval()

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    pin_memory=(device.type == "cuda"),
                    drop_last=(phase == "train"),
                )

                (
                    loss,
                    large_inter,
                    large_union,
                    small_inter_list,
                    small_union_list,
                    mat_loss,
                    refined_nodes,
                    G_loss,
                    reg_epoch_loss,
                ) = run_epoch(
                    model,
                    graph_model,
                    reg_model,
                    dataloader,
                    True,
                    optim,
                    losses,
                    reg_criterion,
                    reg_optimizer,
                    device,
                )

                # Calculate Dice score for large regions
                total_loss = sum(loss for loss in G_loss.values())
                total_loss = total_loss.detach().cpu().numpy()
                phase_losses.append(total_loss)
                large_dice = (
                    2
                    * large_inter.sum().item()
                    / (large_union.sum().item() + large_inter.sum().item())
                )

                # Write metrics to file
                f.write(
                    "{},{},{},{},{},{},{},{},{},{}\n".format(
                        epoch,
                        phase,
                        loss,
                        total_loss,
                        large_dice,
                        time.time() - start_time,
                        large_inter.size,  # Total number of elements in `large_inter`
                        sum(
                            torch.cuda.max_memory_allocated()
                            for i in range(torch.cuda.device_count())
                        ),
                        sum(
                            torch.cuda.max_memory_reserved()
                            for i in range(torch.cuda.device_count())
                        ),
                        batch_size,
                    )
                )
                f.flush()

                if phase == "train":
                    train_losses.append(loss)
                    epoch_metrics["train_loss"] = loss
                    reg_train_losses.append(reg_epoch_loss)
                else:
                    val_losses.append(loss)
                    epoch_metrics["val_loss"] = loss
                    reg_val_losses.append(reg_epoch_loss)

                # Timing info
                elapsed_time = time.time() - start_time
                print(f"  {phase} loss: {loss:.4f} | Time: {elapsed_time:.2f}s")

                # Log metrics and plots to W&B
                epoch_metrics["epoch"] = epoch

                # Plot consolidated train/val loss
                plt.figure(figsize=(10, 5))
                plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
                plt.plot(range(len(val_losses)), val_losses, label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Training and Validation Loss")
                plt.legend()
                wandb.log(epoch_metrics | {"loss_plot": wandb.Image(plt)})
                plt.close()

            scheduler.step()
            reg_scheduler.step()

            # Save checkpoint
            save = {
                "epoch": epoch,
                "state_dict": graph_model.state_dict(),
                "best_loss": bestLoss,
                "graph_loss": total_loss,
                "opt_dict": optim.state_dict(),
                "scheduler_dict": scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "linear_graph_checkpoint.pt"))
            if total_loss < bestLoss:
                torch.save(save, os.path.join(output, "linear_graph_top_best.pt"))
                bestLoss = total_loss

            mlp_save = {
                "epoch": epoch,
                "state_dict": reg_model.state_dict(),
                "best_loss": reg_epoch_loss,
                "opt_dict": reg_optimizer.state_dict(),
            }
            torch.save(mlp_save, os.path.join(output, "cl_reg_checkpoint.pt"))
            if reg_epoch_loss < best_re_loss:
                torch.save(mlp_save, os.path.join(output, "cl_reg_best.pt"))
                best_re_loss = reg_epoch_loss

            # plot graphs per epoch:

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "graph_top_best.pt"))
            graph_model.load_state_dict(checkpoint["state_dict"])
            f.write(
                "Best validation loss {} from epoch {}\n".format(
                    checkpoint["loss"], checkpoint["epoch"]
                )
            )

    # Save videos with segmentation
    if save_video and not all(
        os.path.isfile(os.path.join(output, "videos", f))
        for f in dataloader.dataset.fnames
    ):

        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output, "size"), exist_ok=True)
        echonet.utils.latexify()
        EF_list = []

        with torch.no_grad():
            with open(os.path.join(output, "size.csv"), "w") as g:
                g.write("Filename,Frame,Size,HumanLarge,HumanSmall,ComputerSmall\n")
                for x, (filenames, large_index, small_index, EF), length in tqdm.tqdm(
                    dataloader
                ):
                    # Run segmentation model on blocks of frames one-by-one
                    # The whole concatenated video may be too long to run together

                    EF_list.append(EF)
                    y = np.concatenate(
                        [
                            model(x[i : (i + batch_size), :, :, :].to(device))[0]
                            .detach()
                            .cpu()
                            .numpy()
                            for i in range(0, x.shape[0], batch_size)
                        ]
                    )

                    start = 0
                    x = x.numpy()
                    for i, (filename, offset) in enumerate(zip(filenames, length)):
                        # Extract one video and segmentation predictions
                        video = x[start : (start + offset), ...]
                        logit = y[start : (start + offset), 0, :, :]

                        # Un-normalize video
                        video *= std.reshape(1, 3, 1, 1)
                        video += mean.reshape(1, 3, 1, 1)

                        # Get frames, channels, height, and width
                        f, c, h, w = video.shape  # pylint: disable=W0612
                        assert c == 3

                        # Put two copies of the video side by side
                        video = np.concatenate((video, video), 3)

                        # If a pixel is in the segmentation, saturate blue channel
                        # Leave alone otherwise
                        video[:, 0, :, w:] = np.maximum(
                            255.0 * (logit > 0), video[:, 0, :, w:]
                        )  # pylint: disable=E1111

                        # Add blank canvas under pair of videos
                        video = np.concatenate((video, np.zeros_like(video)), 2)

                        # Compute size of segmentation per frame
                        size = (logit > 0).sum((1, 2))

                        # Identify systole frames with peak detection
                        trim_min = sorted(size)[round(len(size) ** 0.05)]
                        trim_max = sorted(size)[round(len(size) ** 0.95)]
                        trim_range = trim_max - trim_min
                        systole = set(
                            scipy.signal.find_peaks(
                                -size, distance=20, prominence=(0.50 * trim_range)
                            )[0]
                        )

                        # Write sizes and frames to file
                        for frame, s in enumerate(size):
                            g.write(
                                "{},{},{},{},{},{}\n".format(
                                    filename,
                                    frame,
                                    s,
                                    1 if frame == large_index[i] else 0,
                                    1 if frame == small_index[i] else 0,
                                    1 if frame in systole else 0,
                                )
                            )

                        # Plot sizes
                        fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                        plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                        ylim = plt.ylim()
                        for s in systole:
                            plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                        plt.ylim(ylim)
                        plt.title(os.path.splitext(filename)[0])
                        plt.xlabel("Seconds")
                        plt.ylabel("Size (pixels)")
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(
                                output, "size", os.path.splitext(filename)[0] + ".pdf"
                            )
                        )
                        plt.close(fig)

                        # Normalize size to [0, 1]
                        size -= size.min()
                        size = size / size.max()
                        size = 1 - size

                        # Iterate the frames in this video
                        for f, s in enumerate(size):

                            # On all frames, mark a pixel for the size of the frame
                            video[
                                :,
                                :,
                                int(round(115 + 100 * s)),
                                int(round(f / len(size) * 200 + 10)),
                            ] = 255.0

                            if f in systole:
                                # If frame is computer-selected systole, mark with a line
                                video[
                                    :, :, 115:224, int(round(f / len(size) * 200 + 10))
                                ] = 255.0

                            def dash(start, stop, on=10, off=10):
                                buf = []
                                x = start
                                while x < stop:
                                    buf.extend(range(x, x + on))
                                    x += on
                                    x += off
                                buf = np.array(buf)
                                buf = buf[buf < stop]
                                return buf

                            d = dash(115, 224)

                            if f == large_index[i]:
                                # If frame is human-selected diastole, mark with green dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = (
                                    np.array([0, 225, 0]).reshape((1, 3, 1))
                                )
                            if f == small_index[i]:
                                # If frame is human-selected systole, mark with red dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = (
                                    np.array([0, 0, 225]).reshape((1, 3, 1))
                                )

                            # Get pixels for a circle centered on the pixel
                            r, c = skimage.draw.disk(
                                (
                                    int(round(115 + 100 * s)),
                                    int(round(f / len(size) * 200 + 10)),
                                ),
                                4.1,
                            )

                            # On the frame that's being shown, put a circle over the pixel
                            video[f, :, r, c] = 255.0

                        # Rearrange dimensions and save
                        video = video.transpose(1, 0, 2, 3)
                        video = video.astype(np.uint8)
                        echonet.utils.savevideo(
                            os.path.join(output, "videos", filename), video, 50
                        )

                        # Move to next video
                        start += offset


def run_epoch(
    model,
    graph_model,
    reg_model,
    dataloader,
    train,
    optim,
    losses,
    reg_criterion,
    reg_optimizer,
    device,
):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """

    total = 0.0
    n = 0

    pos = 0
    neg = 0
    pos_pix = 0
    neg_pix = 0
    large_inter = 0
    large_union = 0
    small_inter = 0
    small_union = 0
    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []
    EF_list = []
    num_batches = 0
    reg_epoch_loss = 0.0

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for _, (
                large_frame,
                small_frame,
                large_trace,
                small_trace,
                EF,
            ) in dataloader:

                num_batches += 1
                # Count number of pixels in/out of human segmentation
                pos += (large_trace == 1).sum().item()
                pos += (small_trace == 1).sum().item()
                neg += (large_trace == 0).sum().item()
                neg += (small_trace == 0).sum().item()

                # Count number of pixels in/out of computer segmentation
                pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy()
                pos_pix += (small_trace == 1).sum(0).to("cpu").detach().numpy()
                neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
                neg_pix += (small_trace == 0).sum(0).to("cpu").detach().numpy()

                # Run prediction for diastolic frames and compute loss
                large_frame = large_frame.to(device)
                large_trace = large_trace.to(device)
                EF_list.append(EF)

                # y_large_d = model(large_frame)[0]
                y_large_d, features_d = model(large_frame)
                # print(f"input shape {large_frame[0].shape}, decoder{y_large_d[0].shape}, enocder {features_d[0].shape}")

                masks = large_trace.to(device) / 1.0
                masks = masks[:, :1, ...]
                y_large_upsampled = F.interpolate(
                    y_large_d[0], size=(112, 112), mode="bilinear", align_corners=False
                )

                # Now compute the loss with matching sizes
                loss_large = torch.nn.functional.binary_cross_entropy_with_logits(
                    y_large_upsampled[:, 0, :, :], large_trace, reduction="sum"
                )

                # Compute pixel intersection and union between human and computer segmentations
                large_inter += np.logical_and(
                    y_large_upsampled[:, 0, :, :].detach().cpu().numpy() > 0.0,
                    large_trace[:, :, :].detach().cpu().numpy() > 0.0,
                ).sum()
                large_union += np.logical_or(
                    y_large_upsampled[:, 0, :, :].detach().cpu().numpy() > 0.0,
                    large_trace[:, :, :].detach().cpu().numpy() > 0.0,
                ).sum()
                large_inter_list.extend(
                    np.logical_and(
                        y_large_upsampled[:, 0, :, :].detach().cpu().numpy() > 0.0,
                        large_trace[:, :, :].detach().cpu().numpy() > 0.0,
                    ).sum((1, 2))
                )
                large_union_list.extend(
                    np.logical_or(
                        y_large_upsampled[:, 0, :, :].detach().cpu().numpy() > 0.0,
                        large_trace[:, :, :].detach().cpu().numpy() > 0.0,
                    ).sum((1, 2))
                )

                # Run prediction for systolic frames and compute loss
                small_frame = small_frame.to(device)
                small_trace = small_trace.to(device)
                y_small_s, features_s = model(small_frame)

                y_small_upsampled = F.interpolate(
                    y_small_s[0], size=(112, 112), mode="bilinear", align_corners=False
                )
                loss_small = torch.nn.functional.binary_cross_entropy_with_logits(
                    y_small_upsampled[:, 0, :, :], small_trace, reduction="sum"
                )
                # Compute pixel intersection and union between human and computer segmentations
                small_inter += np.logical_and(
                    y_small_upsampled[:, 0, :, :].detach().cpu().numpy() > 0.0,
                    small_trace[:, :, :].detach().cpu().numpy() > 0.0,
                ).sum()
                small_union += np.logical_or(
                    y_small_upsampled[:, 0, :, :].detach().cpu().numpy() > 0.0,
                    small_trace[:, :, :].detach().cpu().numpy() > 0.0,
                ).sum()
                small_inter_list.extend(
                    np.logical_and(
                        y_small_upsampled[:, 0, :, :].detach().cpu().numpy() > 0.0,
                        small_trace[:, :, :].detach().cpu().numpy() > 0.0,
                    ).sum((1, 2))
                )
                small_union_list.extend(
                    np.logical_or(
                        y_small_upsampled[:, 0, :, :].detach().cpu().numpy() > 0.0,
                        small_trace[:, :, :].detach().cpu().numpy() > 0.0,
                    ).sum((1, 2))
                )

                # Take gradient step if training
                loss = (loss_large + loss_small) / 2
                dice_loss = loss
                if train:
                    optim.zero_grad()
                    reg_optimizer.zero_grad()
                refined_nodes, G_loss = graph_model(
                    features_s, y_small_s, features_d, y_large_d
                )

                refined_nodes = refined_nodes.to(device)
                EF = EF.to(device)

                total += dice_loss.item()
                n += large_trace.size(0)
                p = pos / (pos + neg)
                p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)

                if train:

                    reg_loss = train_mlp(reg_model, refined_nodes, EF, reg_criterion)

                    reg_epoch_loss += reg_loss.item()
                    losses.update({"mlp_loss": reg_loss})
                    losses.update(G_loss)
                    total_loss = sum(loss for loss in losses.values())
                    total_loss.backward(retain_graph=False)
                    optim.step()
                    reg_optimizer.step()
                    mat_loss = G_loss["mat_loss_aff"].clone().detach().requires_grad_()
                    # Ensure mat_loss is scalar
                    mat_loss = mat_loss.sum()

                    classification_loss = (
                        G_loss["node_loss"].clone().detach().requires_grad_()
                    )
                    classification_loss = classification_loss.mean()

                    # Show info on process bar
                    pbar.set_postfix_str(
                        "{:.4f} / {:.4f}/ {:.4f}/ {:.4f}".format(
                            dice_loss.item() / large_trace.size(0) / 112 / 112,
                            mat_loss,
                            classification_loss,
                            losses["mlp_loss"],
                        )
                    )
                    pbar.update()

    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)

    print("final numbers:", num_batches)

    # Calculate mean loss for the epoch
    reg_epoch_mae = reg_epoch_loss
    print(f"Epoch MAE: {reg_epoch_mae:.4f}")

    return (
        total / n / 112 / 112,
        large_inter_list,
        large_union_list,
        small_inter_list,
        small_union_list,
        mat_loss,
        refined_nodes,
        G_loss,
        reg_epoch_mae,
    )


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_model(device, pretrained=True):
    """Load and set up the DeepLabV3 model with necessary adjustments."""
    deeplab_model = torchvision.models.segmentation.deeplabv3_resnet50(
        pretrained=pretrained
    )
    deeplab_model.aux_classifier = None
    model = DeepLabFeatureExtractor(deeplab_model)
    model.classifier[-1] = nn.Conv2d(
        model.classifier[-1].in_channels,
        1,
        kernel_size=model.classifier[-1].kernel_size,
    )

    if device.type == "cuda":
        model = nn.DataParallel(model)
    model.to(device)

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


def load_graph_model(device):
    """Initialize and load the graph model."""
    graph_model = GModuleSelfAttention(in_channels=256, num_classes=1, device=device)
    graph_model = graph_model.to(device)
    return graph_model


def load_regressor_model(device):
    """Initialize and load the regressor model."""
    input_size = 128 * 256
    hidden_sizes = [256, 128, 64, 32]
    reg_model = MLP(input_size, hidden_sizes)
    reg_model = reg_model.to(device)
    return reg_model


def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos.

    This function should be used in a dataloader for a dataset that returns
    a video as the first element, along with some (non-zero) tuple of
    targets. Then, the input x is a list of tuples:
      - x[i][0] is the i-th video in the batch
      - x[i][1] are the targets for the i-th video

    This function returns a 3-tuple:
      - The first element is the videos concatenated along the frames
        dimension. This is done so that videos of different lengths can be
        processed together (tensors cannot be "jagged", so we cannot have
        a dimension for video, and another for frames).
      - The second element is contains the targets with no modification.
      - The third element is a list of the lengths of the videos in frames.
    """
    video, target = zip(*x)  # Extract the videos and targets

    i = list(map(lambda t: t.shape[1], video))  # Extract lengths of videos in frames

    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))

    target = zip(*target)

    return video, target, i
