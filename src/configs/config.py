
# -*- coding: utf-8 -*-

"""
Configuration file for the training pipeline.
"""

# Path to the dataset
DATA_PATH = "data/"

# Model parameters
MODEL_NAME = "fasterrcnn_resnet50_fpn"
NUM_CLASSES = 10  # Number of classes + 1 (for background)

# Training parameters
BATCH_SIZE = 4
LEARNING_RATE = 0.005
NUM_EPOCHS = 10
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# Data augmentation parameters
RESIZE_SIZE = (800, 800)

# Callbacks
SAVE_CHECKPOINT = True
CHECKPOINT_PATH = "models/checkpoints/"

# Logging
LOG_PATH = "logs/"
