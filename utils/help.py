import torch
import os
import torchvision
from .dataset import ImageDataset
from torch.utils.data import DataLoader

def get_loaders(train_dir, train_target_dir, val_dir, val_target_dir, test_dir, test_target_dir,
                batch_size=16, train_transform=None, val_transform= None, test_transform = None, 
                num_workers=0, pin_memory=False):
    train_ds = ImageDataset(image_dir=train_dir, target_dir=train_target_dir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    val_ds = ImageDataset(image_dir=val_dir, target_dir=val_target_dir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    test_ds = ImageDataset(image_dir=test_dir, target_dir=test_target_dir, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    return train_loader, val_loader, test_loader

def check_accuracy(loader, model, device='cuda'):
    pass

def save_checkpoints(state, dir="check_points/checkpoints.pth.tar"):
    print("==> Saving checkpoints..")
    torch.save(state, dir)

def load_checkpoints(model):
    print("==> Loading checkpoints...")
    checkpoint = torch.load("check_points/checkpoints.pth.tar", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])