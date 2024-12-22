import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from models.unet import UNet
from torchvision import transforms
from utils.help import (
    setup_logger,
)
from train.generic_trainer import UNetTrainer

def main():
    config = {
        'batch_size': 1,
        'lr': .001,
        'epochs': 3,
        'num_stages': 3,
        'num_unet': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './experiments/sequential/',                                # change the name for each experiments
        'image_height': 256,
        'image_width': 256,
        'num_workers': 2,
        'train_input_dir': "./synthetic_data/train/noisy_images",
        'train_target_dir': "./synthetic_data/train/clean_images",
        'val_input_dir': "./synthetic_data/val/noisy_images",
        'val_target_dir': "./synthetic_data/val/clean_images",
        'test_input_dir': "./synthetic_data/test/noisy_images",
        'test_target_dir': "./synthetic_data/test/clean_images",
    }
    logger = setup_logger(config['save_dir'])
    logger.info("Starting Sequential Training  with config:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")

    system = UNetTrainer(config, logger)

    system.run()


if __name__ == "__main__":
    main()
