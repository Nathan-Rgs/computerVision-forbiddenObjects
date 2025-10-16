
# -*- coding: utf-8 -*-

"""
Logging for the training pipeline.
"""

import logging


def setup_logger(log_path):
    """Set up the logger for the training process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
