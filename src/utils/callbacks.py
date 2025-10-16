
# -*- coding: utf-8 -*-

"""
Callbacks for the training pipeline.
"""

import torch


def save_checkpoint(epoch, model, optimizer, path):
    """Save a checkpoint of the model and optimizer state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
