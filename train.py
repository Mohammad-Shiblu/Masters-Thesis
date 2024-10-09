import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.unet import UNet
from torchvision import transforms
from utils.help import (
    get_loaders
)

# Hyperparameter
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
PIN_MEMORY = False
LOAD_MODEL = False
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# dataset path 
TRAIN_INPUT_DIR = "synthetic_data/train/noisy_image/"
TRAIN_TARGET_DIR = "synthetic_data/train/images/"

VAL_INPUT_DIR = "synthetic_data/val/noisy_image/"
VAL_TARGET_DIR = "synthetic_data/val/images/"

TEST_INPUT_DIR = "synthetic_data/test/noisy_image/"
TEST_TARGET_DIR = "synthetic_data/test/images/"

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (noisy_image, clean_image) in enumerate(loop):
        noisy_image = noisy_image.to(DEVICE)
        clean_image = clean_image.to(DEVICE)

        optimizer.zero_grad()

        # forward pass
        with torch.autocast(DEVICE.type):
            outputs = model(noisy_image)
            loss = loss_fn(outputs, clean_image)

        # backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #update the tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])

    train_loader, val_loader, test_loader = get_loaders(TRAIN_INPUT_DIR, TRAIN_TARGET_DIR, VAL_INPUT_DIR, VAL_TARGET_DIR,
                                                        TEST_INPUT_DIR, TEST_TARGET_DIR, BATCH_SIZE, transform , transform, transform)
    
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)


if __name__ == "__main__":
    main()
