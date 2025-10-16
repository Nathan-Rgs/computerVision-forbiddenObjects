# -*- coding: utf-8 -*-

"""
Main training pipeline.
"""

import torch

from src.configs import config
from src.data.dataset import get_data_loaders
from src.models.model import get_model
from src.utils.callbacks import save_checkpoint
from src.utils.logger import setup_logger

def main():
    """Main function to run the training pipeline."""

    # Setup logger
    setup_logger(config.LOG_PATH)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = get_data_loaders(config.DATA_PATH, config.BATCH_SIZE)

    # Create model
    model = get_model(config.NUM_CLASSES)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop (placeholder)
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        # train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # lr_scheduler.step()
        # evaluate(model, val_loader, device=device)

        if config.SAVE_CHECKPOINT:
            checkpoint_path = f"{config.CHECKPOINT_PATH}checkpoint_epoch_{epoch+1}.pth"
            # save_checkpoint(epoch, model, optimizer, checkpoint_path)

    print("Training pipeline structure created successfully!")


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    """Train the model for one epoch."""
    # Placeholder for the training logic of one epoch
    pass


def evaluate(model, data_loader, device):
    """Evaluate the model."""
    # Placeholder for the evaluation logic
    pass


if __name__ == "__main__":
    # The training is not executed, this is just a placeholder to show how it would be called.
    # main()
    print("To run the training pipeline, uncomment the main() call in train.py")