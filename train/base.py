import torch
from torch.utils.data import DataLoader
import os
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF  # Add this import if needed
from utils.dataset import CtDataset
from utils.metrics import compute_metrics
from torch.utils.data import Subset, Dataset
import matplotlib.pyplot as plt

# Base triner class containing common functions(data loading, model saving/loading, visualization, etc.)
class BaseTrainer:
    def __init__(self, config, logger, test_local=False):
        self.config = config
        self.device = config['device']
        self.test_local = test_local
        self.logger = logger
        self.train_losses = []
        self.val_losses = []
        

    def setup_data(self):
        patch_size = self.config.get('patch_size', None)
        patches_per_image = self.config.get('patches_per_image', 10)
        full_size = (self.config['image_height'], self.config['image_width'])

        # Patch training: no resize on train (patch is already the target size)
        # Full-image training: resize train images to full_size
        train_transform = self.get_transform(target_size=None if patch_size else full_size)
        val_test_transform = self.get_transform(target_size=full_size)

        self.logger.info(
            f"Data mode: {'patch (size=' + str(patch_size) + ', patches_per_image=' + str(patches_per_image) + ')' if patch_size else 'full image (' + str(full_size) + ')'}"
        )

        train_dataset = self.get_dataset(
            transform=train_transform, mode="train",
            patch_size=patch_size,
            patches_per_image=patches_per_image if patch_size else None
        )
        val_dataset = self.get_dataset(transform=val_test_transform, mode="val")
        test_dataset = self.get_dataset(transform=val_test_transform, mode="test")

        if self.test_local:
            train_dataset = Subset(train_dataset, range(20))
            val_dataset = Subset(val_dataset, range(2))
            test_dataset = Subset(test_dataset, range(2))

        self.train_loader = self.get_loader(train_dataset, shuffle=True)
        self.val_loader = self.get_loader(val_dataset, shuffle=False)
        self.test_loader = self.get_loader(test_dataset, shuffle=False)
        self.logger.info("--------------data loading completed----------")


    def get_transform(self, target_size=None):
        """
        Build a transform pipeline.
        - target_size=None : ToTensor only (use for patch training — patches need no resize)
        - target_size=(H,W): ToTensor + Resize (use for full-image training/val/test)
        """
        ops = [transforms.ToTensor()]
        if target_size is not None:
            ops.append(transforms.Resize(target_size))
        return transforms.Compose(ops)
    
    def get_dataset(self, transform, mode, patch_size= None, patches_per_image=None):
        return CtDataset(self.config, transform=transform, mode=mode, patch_size=patch_size, patches_per_image=patches_per_image)
    
    def get_loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=shuffle,
            num_workers=self.config["num_workers"]
        )

    def set_train_epoch(self, epoch):
        """Notify the training dataset of the current epoch for varied patch sampling."""
        ds = self.train_loader.dataset
        if isinstance(ds, Subset):
            ds = ds.dataset
        if hasattr(ds, 'set_epoch'):
            ds.set_epoch(epoch)

    def setup_models(self):
        raise NotImplementedError("setup_models() needs to be implemented in the subclass.")

  
    
    def train(self):
        raise NotImplementedError("train_stage() needs to be implemented in the subclass.")
    
    def validate(self):
        raise NotImplementedError("validate_stage() needs to be implemented in the subclass.")
    
    def test(self):
        raise NotImplementedError("test_system() needs to be implemented in the subclass.")
    
    def visualize_results(self, num_samples=5):
        raise NotImplementedError("visualize_results() needs to be implemented in the subclass.")
    
    def save_model(self):
        file_path = os.path.join(self.config["checkpoints_dir"], self.config["model"])
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        torch.save(self.model.state_dict(), os.path.join(file_path, f"checkpoints_{self.config['test_no']}.pth"))
    
    def load_model(self):
        file_path = os.path.join(self.config["checkpoints_dir"], self.config["model"], f"checkpoints_{self.config['test_no']}.pth")
        self.model.load_state_dict(torch.load(file_path, weights_only=True))
    
    # save test images
    def save_image(self, fig_name, input, pred, target):
        # save the test images of input, pred, output
        input, pred, target = input.cpu().numpy(), pred.cpu().numpy(), target.cpu().numpy()
        ori_metrics, pred_metrics = compute_metrics(input, pred, target) # return (rmse, psnr, ssim) for both input and pred
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        # input noisy image
        ax[0].imshow(input.squeeze(), cmap='gray')
        ax[0].set_title('Noisy_image', fontsize=28)
        ax[0].set_xlabel(f"PSNR: {ori_metrics[1]:.3f}\nSSIM: {ori_metrics[2]:.3f}", fontsize=20)
        # pred 
        ax[1].imshow(pred.squeeze(), cmap='gray')
        ax[1].set_title('Denoised Output', fontsize=28)
        ax[1].set_xlabel(f"PSNR: {pred_metrics[1]:.3f}\nSSIM: {pred_metrics[2]:.3f}", fontsize=20)
        # target
        ax[2].imshow(target.squeeze(), cmap="gray")
        ax[2].set_title("Full dose", fontsize=24)

        file_path = os.path.join(self.config["output_dir"], self.config['model'], "fig", self.config['test_no'], "results")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        
        f.savefig(os.path.join(file_path, f"results_{fig_name}.png"))
        plt.close()

    # plot the loss curves across epochs
    def plot_loss_curves(self): 
        # plot the loss curves
        epochs = range(1 , len(self.train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')

        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        file_path = os.path.join(self.config["output_dir"], self.config['model'], "fig", self.config['test_no'])
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        plt.savefig(os.path.join(file_path, f"loss_curves{self.config['test_no']}.png"))
        


    





