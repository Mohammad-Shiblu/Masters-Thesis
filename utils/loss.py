import torch
import torch.nn as nn
from torch.nn import AvgPool2d
import lpips
import os
import torch.nn.functional as F


def make_loss_fns(logger, num_stages, *, alpha=0.84, beta=0.16, device='cuda', use_l1_in_combined=True):

    if num_stages < 1:
        raise ValueError("num_stages must be >= 1")

    losses = []
    losses.append(nn.L1Loss().to(device)) # first stage loss
    logger.info(f"Stage 0 loss: {losses[-1].__class__.__name__}")
    
    if num_stages >= 2:
        # Middle stages (if any): L1
        for i in range(1, num_stages):
            loss = GrayscaleLPIPSLoss(net='vgg').to(device)
            losses.append(loss)
            logger.info(f"Stage {i} loss: {losses[-1].__class__.__name__}")
        # Final stage: CombinedLoss
        # losses.append(CombinedLoss(alpha=alpha, beta=beta, l1_loss=use_l1_in_combined))
    return losses

def log_loss_scales(logger, losses, pred, target):
    """
    Helper function to log the scale of each loss for debugging
    Args:
        logger: Logger instance
        losses: List of loss functions
        pred: Predicted tensor
        target: Target tensor
    """
    logger.info("=" * 50)
    logger.info("Loss Scales Analysis:")
    logger.info("=" * 50)
    
    with torch.no_grad():
        for i, loss_fn in enumerate(losses):
            loss_val = loss_fn(pred[i], target)
            logger.info(f"Stage {i} ({loss_fn.__class__.__name__}): {loss_val.item():.6f}")
    
    logger.info("=" * 50)

# mean square loss function using mean square loss and the penalty (Not imrproved the output than normal )
def compute_loss(pred, target):
    mse_loss = F.mse_loss(pred, target)
    range_penalty = torch.mean(F.relu(pred -1.0) + F.relu(-pred))
    return mse_loss + 0.05 * range_penalty

class CombinedLoss(nn.Module):
    """
    Combined loss function for CT denoising
    Combines SSIM loss with L1/L2 loss for better convergence
    """
    def __init__(self, alpha=0.7, beta=0.3, l1_loss=True):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # Weight for SSIM loss
        self.beta = beta    # Weight for pixel-wise loss
        self.ssim_loss = SSIMLoss()
        
        if l1_loss:
            self.pixel_loss = nn.L1Loss()
        else:
            self.pixel_loss = nn.MSELoss()
            
    def forward(self, pred, target):
        ssim_loss_val = self.ssim_loss(pred, target)
        pixel_loss_val = self.pixel_loss(pred, target)
        
        total_loss = self.alpha * ssim_loss_val + self.beta * pixel_loss_val
        return total_loss

# calculate LPIPS loss for grayscale image in [0, 1]
class GrayscaleLPIPSLoss(nn.Module):
    def __init__(self, net= "alex"):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net)
        for param in self.lpips.parameters():
            param.requires_grad = False

    @staticmethod
    def prep(x):
        # if gray scale repeat channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        # [0, 1] -> [-1, 1]
        return x * 2.0 - 1.0  
    
    def forward(self, pred, target):
        pred = self.prep(pred.clamp(0.0, 1.0))
        target = self.prep(target.clamp(0.0, 1.0))
        
        d = self.lpips(pred, target)
        return d.mean()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, k1=0.01, k2=0.03, channel=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.k1 = k1
        self.k2 = k2

        self.window = self.create_window(window_size, sigma, channel)

    def create_window(self, window_size, sigma, channel):
        """Create Gaussian window """
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2

        # 1D gaussian kernel
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        # Create 2D Gaussian kernel
        kernel = g.unsqueeze(1) * g.unsqueeze(0) # outer product
        kernel = kernel.unsqueeze(0).unsqueeze(1) # (1, 1, H, W)

        return kernel
    
    def gaussian_filter(self, input, window):
        padding = self.window_size // 2
        input_padded = F.pad(input, (padding, padding, padding, padding), mode='reflect')
        out= F.conv2d(input_padded, window, groups=self.channel)
        return out

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        if img1.device != window.device:
            window = window.to(img1.device)
        mu1 = self.gaussian_filter(img1, window)
        mu2 = self.gaussian_filter(img2, window)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.gaussian_filter(img1 * img1, window) - mu1_sq
        sigma2_sq = self.gaussian_filter(img2 * img2, window) -mu2_sq
        sigma12 = self.gaussian_filter(img1 * img2, window) - mu1_mu2

        C1 = (self.k1) ** 2
        C2 = (self.k2) ** 2

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / (denominator + 1e-8)  # Add epsilon for numerical stability
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
        
    def forward(self, img1, img2):
        """
        Forward pass - returns SSIM loss (1 - SSIM)
        Args:
            img1: Predicted/denoised image [B, C, H, W]
            img2: Ground truth/clean image [B, C, H, W]
        Returns:
            SSIM loss value (lower is better)
        """
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = self.create_window(self.window_size, 1.5, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
            
        ssim_value = self.ssim(img1, img2, window, self.window_size, channel, size_average=True)
        return 1 - ssim_value  # Return loss (1 - SSIM)
    


if __name__ == '__main__':
    pred = torch.rand(2, 1, 32, 32, requires_grad=True)  # Random predictions
    target = torch.rand(2, 1, 32, 32)
    criterion = GrayscaleLPIPSLoss(net="vgg")
    loss = criterion(pred, target)
    loss.backward()

    print("Predictions:", pred.data)
    print("Gradients: ", pred.grad)  # value should be non zero everywhere