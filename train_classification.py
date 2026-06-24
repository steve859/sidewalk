import argparse
import os
import csv
from datetime import datetime

import torch
import torch.nn as nn

from src.models.classifier import create_model, ENCODER_ZOO
from src.losses.losses import FocalLoss, AsymmetricLoss
from src.label_map import NUM_CLASSES


def get_loss_fn(loss_name, device):
    loss_name = loss_name.lower()

    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_name == "weighted_bce":
        # Temporary placeholder.
        # Later, calculate this from real training label distribution.
        pos_weight = torch.ones(NUM_CLASSES) * 2.0
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    if loss_name == "focal":
        return FocalLoss(alpha=1.0, gamma=2.0, reduction="mean")

    if loss_name == "asl":
        return AsymmetricLoss(
            gamma_neg=4,
            gamma_pos=1,
            clip=0.05,
        )

    raise ValueError(
        f"Unknown loss: {loss_name}. "
        "Available losses: bce, weighted_bce, focal, asl"
    )


def save_checkpoint(
    save_path,
    model,
    optimizer,
    epoch,
    best_loss,
    args,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
        "encoder_name": args.encoder_name,
        "aggregation": args.aggregation,
        "loss": args.loss,
        "num_classes": NUM_CLASSES,
    }

    torch.save(checkpoint, save_path)


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    advanced_aug=None,
):
    """
    Real training epoch.

    For now, if dataloader is None, this returns None.
    Later, replace train_loader = None with real DataLoader.
    """

    if dataloader is None:
        return None

    model.train()
    total_loss = 0.0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.float().to(device)

        if advanced_aug is not None:
            images, targets = advanced_aug(images, targets)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def run_dummy_epoch(epoch):
    """
    Temporary dummy loss for testing training-script logic
    before the real dataset is ready.
    """

    return 1.0 / (epoch + 1)


def write_experiment_log(args, best_loss):
    log_file = os.path.join(args.save_dir, "experiments_log.csv")
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "time",
                "note",
                "encoder_name",
                "loss",
                "aggregation",
                "advanced_aug",
                "pretrained",
                "best_loss",
            ])

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        writer.writerow([
            current_time,
            args.note,
            args.encoder_name,
            args.loss,
            args.aggregation,
            args.advanced_aug,
            args.pretrained,
            f"{best_loss:.4f}",
        ])

    print(f"Experiment log saved to: {log_file}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    print("\n--- Sidewalk Encroachment Classification Training ---")
    print(f"Device: {device}")
    print(f"Save directory: {args.save_dir}")
    print(f"Encoder: {args.encoder_name}")
    print(f"Timm model: {ENCODER_ZOO[args.encoder_name]}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Loss: {args.loss}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Advanced augmentation: {args.advanced_aug}")
    print(f"Number of classes: {NUM_CLASSES}")

    model = create_model(
        encoder_name=args.encoder_name,
        aggregation=args.aggregation,
        num_classes=NUM_CLASSES,
        pretrained=args.pretrained,
    )

    model = model.to(device)

    criterion = get_loss_fn(args.loss, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    advanced_aug = None

    if args.advanced_aug in ["mixup", "cutmix"]:
        raise NotImplementedError(
            "Mixup/CutMix will be added after augmentations.py is finalized."
        )

    train_loader = None

    best_loss = float("inf")

    print("\nTraining loop started.")
    print("Dataset is not ready yet, so this script is running in dummy mode.\n")

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            advanced_aug=advanced_aug,
        )

        if train_loss is None:
            train_loss = run_dummy_epoch(epoch)

        last_model_path = os.path.join(args.save_dir, "last_model.pth")

        save_checkpoint(
            save_path=last_model_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            best_loss=best_loss,
            args=args,
        )

        if train_loss < best_loss:
            best_loss = train_loss

            best_model_path = os.path.join(args.save_dir, "best_model.pth")

            save_checkpoint(
                save_path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_loss=best_loss,
                args=args,
            )

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"loss: {train_loss:.4f} | "
                "best improved -> saved best_model.pth"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"loss: {train_loss:.4f} | "
                "saved last_model.pth"
            )

    write_experiment_log(args, best_loss)

    print("\nTraining script finished.")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved in: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints and logs.",
    )

    parser.add_argument(
        "--note",
        type=str,
        default="No note",
        help="Experiment note.",
    )

    parser.add_argument(
        "--encoder_name",
        type=str,
        default="resnet50",
        choices=list(ENCODER_ZOO.keys()),
        help="Encoder name from ENCODER_ZOO.",
    )

    parser.add_argument(
        "--aggregation",
        type=str,
        default="gap",
        choices=["gap", "attention", "gated_attention", "cls_token"],
        help="Feature aggregation method.",
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="bce",
        choices=["bce", "weighted_bce", "focal", "asl"],
        help="Loss function.",
    )

    parser.add_argument(
        "--advanced_aug",
        type=str,
        default="none",
        choices=["none", "mixup", "cutmix"],
        help="Advanced augmentation method.",
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights. Default is False.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs.",
    )

    args = parser.parse_args()
    main(args)