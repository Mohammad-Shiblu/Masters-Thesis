import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.transforms as transforms
from pytorch_msssim import ssim
import numpy as np
from utils.dataset import ImageDataset


class BaseTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.device = config['device']
        self.logger = logger
        self.setup_data()
        self.setup_models()

    def setup_data(self):
        transform = transforms.Compose([
            transforms.Resize((self.config['image_height'], self.config['image_width'])),
            transforms.ToTensor()
        ])
        train_dataset = ImageDataset(
            image_dir = self.config['train_input_dir'],
            target_dir= self.config['train_target_dir'],
            transform= transform,
        )
        val_dataset = ImageDataset(
            image_dir = self.config['val_input_dir'],
            target_dir= self.config['val_target_dir'],
            transform= transform,
        )
        test_dataset = ImageDataset(
            image_dir = self.config['test_input_dir'],
            target_dir= self.config['test_target_dir'],
            transform= transform,
        )
        # Data loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )

    def setup_models(self):
        raise NotImplementedError("setup_models() needs to be implemented in the subclass.")

    @staticmethod
    def normalize(var):
        var = (var - var.min()) / (var.max() - var.min())
        return var
    @staticmethod
    def normalize_to_target(output, target):
        target_mean = target.mean(dim=(1, 2, 3), keepdim=True)
        target_std = target.std(dim=(1, 2, 3), keepdim=True)
        output_mean = output.mean(dim=(1, 2, 3), keepdim=True)
        output_std = output.std(dim=(1, 2, 3), keepdim=True)
        
        normalized_output = (output - output_mean) / (output_std + 1e-8)  
        normalized_output = normalized_output * target_std + target_mean  

        # normalized_output = 2 * normalized_output - 1

        return normalized_output

    @staticmethod
    def calculate_metrics(pred, target):
        """calculate PSNR and SSIM metrics."""
        # Normalize the values to [0, 1] ranges to metric calculation
        pred = BaseTrainer.normalize(pred)
        target = BaseTrainer.normalize(target)

        # Calculate PSNR
        mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])  # Mean squared error per image
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        psnr = psnr.mean()

        # Calculate SSIM 
        ssim_val = ssim(pred, target, data_range=1.0, size_average=False)

        return psnr.item(), ssim_val.item()
    
    def train_stage(self, stage_num):
        raise NotImplementedError("setup_models() needs to be implemented in the subclass.")
    
    def validate_stage(self, stage_num):
        raise NotImplementedError("setup_models() needs to be implemented in the subclass.")
    
    def test_system(self):
        raise NotImplementedError("setup_models() needs to be implemented in the subclass.")
    
    def visualize_results(self, num_smaples=5):
        self.model.eval()

        noisy_imgs_list = []
        clean_imgs_list = []
        test_iter = iter(self.test_loader)
        for _ in range(num_smaples):
            try: 
                noisy_img, clean_img = next(test_iter)
            except StopIteration:
                # Reset the iterator if there are fewer samples than requested
                test_iter = iter(self.test_loader)
                noisy_img, clean_img = next(test_iter)
            # Move images to the device and store them in lists
            noisy_imgs_list.append(noisy_img.to(self.device))
            clean_imgs_list.append(clean_img.to(self.device))
