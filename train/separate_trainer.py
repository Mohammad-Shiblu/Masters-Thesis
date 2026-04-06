import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from train.base import BaseTrainer
from models.unet import UNet
from models.munet import CascadedUNet
from utils.dataset import CtDataset
from utils.loss import make_loss_fns, GrayscaleLPIPSLoss, SSIMLoss
from utils.metrics import compute_psnr, compute_ssim, compute_rmse
from utils.help import EarlyStopping
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

class ProgressiveCascadedTrainer(BaseTrainer):
    """
    Progressive Cascaded U-Net Trainer with:
    1. Non-overlapping data splits per stage
    2. Progressive noise reduction targets
    3. Different loss functions per stage
    4. Residual learning between stages
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_stages = self.config['num_unets']
        
        # Log training strategy
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Progressive Cascaded U-Net Trainer - Test {self.config['test_no']}")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Number of stages: {self.num_stages}")
        self.logger.info(f"Training strategy: Progressive noise reduction + Non-overlapping data")
        self.logger.info(f"Patch size: {self.config.get('patch_size', 128)}")
        self.logger.info(f"Patches per image: {self.config.get('patches_per_image', 10)}")
        
        # Setup data with stage-specific splits
        self.setup_progressive_data()
        
        # Setup models for each stage
        self.setup_models()
        
        # Setup stage-specific loss functions
        self.setup_stage_losses()
        
        # Setup optimizers and schedulers for each stage
        self.setup_optimizers()
        
        # Track metrics
        self.stage_train_losses = [[] for _ in range(self.num_stages)]
        self.stage_val_losses = [[] for _ in range(self.num_stages)]
        self.stage_best_epochs = [0] * self.num_stages
    
    def setup_progressive_data(self):
        """
        Setup data with non-overlapping splits for each stage.
        Each stage gets a different subset of the training data.
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("Setting up progressive data splits...")
        self.logger.info("="*80)

        patch_size = self.config.get('patch_size', None)
        patches_per_image = self.config.get('patches_per_image', 10)
        full_size = (self.config['image_height'], self.config['image_width'])

        # Reuse base get_transform: no resize for patches, resize for full images
        train_transform = self.get_transform(target_size=None if patch_size else full_size)
        val_test_transform = self.get_transform(target_size=full_size)

        self.logger.info(
            f"Data mode: {'patch (size=' + str(patch_size) + ', patches_per_image=' + str(patches_per_image) + ')' if patch_size else 'full image (' + str(full_size) + ')'}"
        )

        # Create full training dataset with patches
        full_train_dataset = CtDataset(
            self.config,
            transform=train_transform,
            mode='train',
            patch_size=patch_size,
            patches_per_image=patches_per_image if patch_size else None
        )

        # Create validation and test datasets (full images, no patches)
        val_dataset = CtDataset(
            self.config,
            transform=val_test_transform,
            mode='val',
            patch_size=None  # Full images for validation
        )

        test_dataset = CtDataset(
            self.config,
            transform=val_test_transform,
            mode='test',
            patch_size=None  # Full images for testing
        )
        
        # Apply test_local if needed
        if self.test_local:
            full_train_dataset = Subset(full_train_dataset, range(20))
            val_dataset = Subset(val_dataset, range(2))
            test_dataset = Subset(test_dataset, range(2))

        # Keep a direct reference so we can call set_epoch() each training epoch
        self.full_train_dataset = full_train_dataset
        
        # Split training data into non-overlapping subsets for each stage
        total_len = len(full_train_dataset)
        stage_size = total_len // self.num_stages
        
        # Shuffle indices
        indices = list(range(total_len))
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        # Create stage-specific data loaders
        self.stage_train_loaders = []
        self.stage_val_loaders = []
        
        for stage in range(self.num_stages):
            # Get indices for this stage
            start_idx = stage * stage_size
            end_idx = start_idx + stage_size if stage < self.num_stages - 1 else total_len
            stage_indices = indices[start_idx:end_idx]
            
            # Create subset for this stage
            stage_dataset = Subset(full_train_dataset, stage_indices)
            
            # Split into train/val (90/10 for each stage)
            train_size = int(0.9 * len(stage_dataset))
            val_size = len(stage_dataset) - train_size
            stage_train, stage_val = torch.utils.data.random_split(
                stage_dataset, [train_size, val_size]
            )
            
            # Create loaders
            train_loader = DataLoader(
                stage_train,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers']
            )
            
            val_loader = DataLoader(
                stage_val,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers']
            )
            
            self.stage_train_loaders.append(train_loader)
            self.stage_val_loaders.append(val_loader)
            
            self.logger.info(
                f"Stage {stage}: {len(stage_train)} train patches, "
                f"{len(stage_val)} val patches"
            )
        
        # Global validation and test loaders (full images and with batch size of 1)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Full images, batch_size=1
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Full images, batch_size=1
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        self.logger.info(f"Global validation: {len(val_dataset)} full images")
        self.logger.info(f"Global test: {len(test_dataset)} full images")
        self.logger.info("="*80 + "\n")
    
    def setup_models(self):
        """Setup individual U-Net models for each stage."""
        self.logger.info("Setting up stage models...")
        
        self.models = []
        for stage in range(self.config['num_unets']):
            # Using smp's U-Net implementation
            # model = smp.Unet(
            #     encoder_name=self.config['model_params']['encoder_name'],
            #     decoder_channels=[64, 32, 16],    
            #     decoder_use_batchnorm=True,       
            #     encoder_depth=3,                  
            #     encoder_weights=self.config['model_params']['encoder_weights'],       
            #     in_channels=self.config['model_params']['in_channels'],                   
            #     classes=self.config['model_params']['classes']                     
            # ).to(self.device)
            model = UNet(in_channels=1, out_channels=1, features=[16, 32, 64]).to(self.device)
            self.models.append(model)
            
            num_params = sum(p.numel() for p in model.parameters())
            self.logger.info(f"Stage {stage} U-Net: {num_params:,} parameters")
        
        self.logger.info(f"Total parameters: {sum(p.numel() for m in self.models for p in m.parameters()):,}\n")
    
    def setup_stage_losses(self):
        """Setup different loss functions for each stage."""
        self.logger.info("Setting up stage-specific loss functions...")
        
        self.stage_losses = []
        
        for stage in range(self.num_stages):
            if stage == 0:
                # Stage 0: L1 loss for initial denoising
                # loss_fn = nn.L1Loss().to(self.device)
                # loss_name = "L1Loss"
                loss_fn = GrayscaleLPIPSLoss(net='alex').to(self.device)
                loss_name = "LPIPS"
            elif stage == self.num_stages - 1:
                # Final stage: Perceptual loss (LPIPS)
                loss_fn = GrayscaleLPIPSLoss(net='alex').to(self.device)
                loss_name = "LPIPS"
                
            else:
                # Middle stages: SSIM loss
                # loss_fn = SSIMLoss().to(self.device)
                # loss_name = "SSIMLoss"
                loss_fn = GrayscaleLPIPSLoss(net='alex').to(self.device)
                loss_name = "LPIPS"
            
            self.stage_losses.append(loss_fn)
            self.logger.info(f"Stage {stage}: {loss_name}")
        
        self.logger.info("")
    
    def setup_optimizers(self):
        """Setup optimizers and schedulers for each stage."""
        self.optimizers = []
        self.schedulers = []
        
        for stage in range(self.num_stages):
            optimizer = optim.AdamW(
                self.models[stage].parameters(),
                lr=self.config['lr']
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
            
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
    
    def preprocess_with_previous_stages(self, inputs, up_to_stage):
        """
        Process inputs through all previous stages.
        
        Args:
            inputs: Input tensor
            up_to_stage: Process through stages 0 to up_to_stage-1
        
        Returns:
            Processed tensor
        """
        if up_to_stage == 0:
            return inputs
        
        with torch.no_grad():
            stage_input = inputs
            for stage in range(up_to_stage):
                self.models[stage].eval()
                stage_input = self.models[stage](stage_input)
                stage_input = torch.clamp(stage_input, 0.0, 1.0)
        
        return stage_input
    
    def train_single_epoch(self, stage_num, epoch):
        """Train a single stage for one epoch."""
        model = self.models[stage_num]
        optimizer = self.optimizers[stage_num]
        criterion = self.stage_losses[stage_num]
        
        model.train()
        running_loss = 0.0
        
        train_loader = self.stage_train_loaders[stage_num]
        loop = tqdm(train_loader, desc=f"Stage {stage_num} Epoch {epoch:03d}")
        
        for inputs, targets in loop:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Preprocess with previous stages
            stage_inputs = self.preprocess_with_previous_stages(inputs, stage_num)
            
            # Forward pass through current stage
            optimizer.zero_grad()
            outputs = model(stage_inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Handle LPIPS loss (might return tensor with shape)
            if loss.dim() > 0:
                loss = loss.mean()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / len(train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate_single_epoch(self, stage_num, epoch):
        """Validate a single stage for one epoch."""
        model = self.models[stage_num]
        criterion = self.stage_losses[stage_num]
        
        model.eval()
        running_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        
        val_loader = self.stage_val_loaders[stage_num]
        
        for inputs, targets in val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Preprocess with previous stages
            stage_inputs = self.preprocess_with_previous_stages(inputs, stage_num)
            
            # Forward pass
            outputs = model(stage_inputs)
            outputs = torch.clamp(outputs, 0.0, 1.0)
            
            # Compute loss
            loss = criterion(outputs, targets)
            if loss.dim() > 0:
                loss = loss.mean()
            
            # Compute metrics
            psnr = compute_psnr(outputs, targets)
            ssim = compute_ssim(outputs, targets)
            
            running_loss += loss.item()
            running_psnr += psnr
            running_ssim += ssim
        
        n = len(val_loader)
        return {
            'loss': running_loss / n,
            'psnr': running_psnr / n,
            'ssim': running_ssim / n
        }
    
    def train_stage(self, stage_num):
        """Train a single stage for all epochs."""
        self.logger.info("\n" + "="*80)
        self.logger.info(f"Training Stage {stage_num}")
        self.logger.info("="*80)
        
        early_stopping = EarlyStopping(patience=self.config['patience'], min_delta=0.0)
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            # Vary patch positions each epoch
            ds = self.full_train_dataset
            if isinstance(ds, Subset):
                ds = ds.dataset
            if hasattr(ds, 'set_epoch'):
                ds.set_epoch(epoch)

            # Train
            train_loss = self.train_single_epoch(stage_num, epoch)
            
            # Validate
            val_metrics = self.validate_single_epoch(stage_num, epoch)
            
            # Update scheduler
            self.schedulers[stage_num].step(val_metrics['loss'])
            
            # Log
            self.logger.info(
                f"Stage {stage_num} Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val PSNR: {val_metrics['psnr']:.2f} | "
                f"Val SSIM: {val_metrics['ssim']:.4f}"
            )
            
            # Save losses for loss curves
            self.stage_train_losses[stage_num].append(train_loss)
            self.stage_val_losses[stage_num].append(val_metrics['loss'])
            
            # Early stopping
            early_stopping.check_early_stop(val_metrics['loss'])
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_stage_model(stage_num)
                self.stage_best_epochs[stage_num] =  epoch
                self.logger.info(f"✓ Best model saved for stage {stage_num}")
            
            if early_stopping.stop_training:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Load best model for this stage
        self.load_stage_model(stage_num)
        self.logger.info(f"Loaded best model for stage {stage_num}")
        self.logger.info("="*80 + "\n")
    
    def train(self):
        """Train all stages sequentially."""
        self.logger.info("\n" + "#"*80)
        self.logger.info("STARTING PROGRESSIVE CASCADED TRAINING")
        self.logger.info("#"*80 + "\n")
        
        for stage in range(self.num_stages):
            self.train_stage(stage)
            
            # Validate on full validation set after each stage
            self.logger.info(f"\nValidating full cascade up to stage {stage}...")
            cascade_metrics = self.validate_full_cascade(up_to_stage=stage+1)
            
            self.logger.info(
                f"Cascade (Stages 0-{stage}) on full validation set: "
                f"PSNR: {cascade_metrics['psnr']:.2f} | "
                f"SSIM: {cascade_metrics['ssim']:.4f}"
            )
        
        self.logger.info("\n" + "#"*80)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("#"*80 + "\n")
        
        # Plot loss curves
        self.plot_stage_loss_curves()
    
    @torch.no_grad()
    def validate_full_cascade(self, up_to_stage=None):
        """
        Validate the full cascade on validation set.
        
        Args:
            up_to_stage: Validate using stages 0 to up_to_stage-1 (None = all stages)
        """
        if up_to_stage is None:
            up_to_stage = self.num_stages
        
        for model in self.models[:up_to_stage]:
            model.eval()
        
        running_psnr = 0.0
        running_ssim = 0.0
        running_rmse = 0.0
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Process through cascade
            stage_output = inputs
            for stage in range(up_to_stage):
                stage_output = self.models[stage](stage_output)
                stage_output = torch.clamp(stage_output, 0.0, 1.0)
            
            # Compute metrics
            running_psnr += compute_psnr(stage_output, targets)
            running_ssim += compute_ssim(stage_output, targets)
            running_rmse += compute_rmse(stage_output, targets)
        
        n = len(self.val_loader)
        return {
            'psnr': running_psnr / n,
            'ssim': running_ssim / n,
            'rmse': running_rmse / n
        }
    
    @torch.no_grad()
    def test(self):
        """Test the full cascade on test set."""
        self.logger.info("\n" + "="*80)
        self.logger.info("TESTING FULL CASCADE")
        self.logger.info("="*80)
        
        # Load all stage models
        for stage in range(self.num_stages):
            self.load_stage_model(stage)
            self.models[stage].eval()
        
        # Metrics tracking
        baseline_psnr = baseline_ssim = baseline_rmse = 0.0
        stage_psnr = [0.0] * self.num_stages
        stage_ssim = [0.0] * self.num_stages
        stage_rmse = [0.0] * self.num_stages
        
        # randomly picking 10 images for visualisation (fixed seed for reproducibility)
        total_test = len(self.test_loader)
        rng = np.random.default_rng(seed=42)
        save_indices = set(rng.choice(total_test, size=min(10, total_test), replace=False).tolist())
        self.logger.info(f"Saving visualizations for image indices: {sorted(save_indices)}")

        img_idx = 0
        
        for inputs, targets in tqdm(self.test_loader, desc="Testing"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Baseline metrics (noisy input)
            baseline_psnr += compute_psnr(inputs, targets)
            baseline_ssim += compute_ssim(inputs, targets)
            baseline_rmse += compute_rmse(inputs, targets)
            
            # Process through cascade
            stage_outputs = []
            stage_output = inputs
            
            for stage in range(self.num_stages):
                stage_output = self.models[stage](stage_output)
                stage_output = torch.clamp(stage_output, 0.0, 1.0)
                stage_outputs.append(stage_output)
                
                # Compute metrics for this stage
                stage_psnr[stage] += compute_psnr(stage_output, targets)
                stage_ssim[stage] += compute_ssim(stage_output, targets)
                stage_rmse[stage] += compute_rmse(stage_output, targets)
            
            # Save visualization for randomly selcted indices
            if img_idx in save_indices:  # Save first 10 images
                self.save_cascade_image(img_idx, inputs, stage_outputs, targets)
            img_idx += 1
        
        # Compute averages
        n = len(self.test_loader)
        baseline_psnr /= n
        baseline_ssim /= n
        baseline_rmse /= n
        
        stage_psnr = [p / n for p in stage_psnr]
        stage_ssim = [s / n for s in stage_ssim]
        stage_rmse = [r / n for r in stage_rmse]
        
        # Log results
        self.logger.info(f"\n[Baseline] PSNR: {baseline_psnr:.2f} | SSIM: {baseline_ssim:.4f} | RMSE: {baseline_rmse:.4f}")
        
        for stage in range(self.num_stages):
            psnr_gain = stage_psnr[stage] - baseline_psnr
            ssim_gain = stage_ssim[stage] - baseline_ssim
            rmse_gain = baseline_rmse - stage_rmse[stage]
            
            if stage > 0:
                psnr_gain_prev = stage_psnr[stage] - stage_psnr[stage-1]
                ssim_gain_prev = stage_ssim[stage] - stage_ssim[stage-1]
                rmse_gain_prev = stage_rmse[stage-1] - stage_rmse[stage]
                
                self.logger.info(
                    f"[Stage {stage}] "
                    f"PSNR: {stage_psnr[stage]:.2f} (+{psnr_gain:.2f} vs base, +{psnr_gain_prev:.2f} vs prev) | "
                    f"SSIM: {stage_ssim[stage]:.4f} (+{ssim_gain:.4f} vs base, +{ssim_gain_prev:.4f} vs prev) | "
                    f"RMSE: {stage_rmse[stage]:.4f} ({rmse_gain:.4f}↓ vs base, {rmse_gain_prev:.4f}↓ vs prev)"
                )
            else:
                self.logger.info(
                    f"[Stage {stage}] "
                    f"PSNR: {stage_psnr[stage]:.2f} (+{psnr_gain:.2f} vs base) | "
                    f"SSIM: {stage_ssim[stage]:.4f} (+{ssim_gain:.4f} vs base) | "
                    f"RMSE: {stage_rmse[stage]:.4f} ({rmse_gain:.4f}↓ vs base)"
                )
        
        self.logger.info("="*80 + "\n")
        # Store results
        self.test_results = {
            "baseline": {
                "psnr": baseline_psnr.cpu().item() if hasattr(baseline_psnr, 'cpu') else float(baseline_psnr),
                "ssim": baseline_ssim.cpu().item() if hasattr(baseline_ssim, 'cpu') else float(baseline_ssim),
                "rmse": baseline_rmse.cpu().item() if hasattr(baseline_rmse, 'cpu') else float(baseline_rmse),
            },
            "stages": {
                "psnr": [v.cpu().item() if hasattr(v, 'cpu') else float(v) for v in stage_psnr],
                "ssim": [v.cpu().item() if hasattr(v, 'cpu') else float(v) for v in stage_ssim],
                "rmse": [v.cpu().item() if hasattr(v, 'cpu') else float(v) for v in stage_rmse],
            }
}

        # Plot improvement
        self.plot_stage_improvement()
        self.logger.info(f"\nStage improvement plot has been saved")
        
    
    def save_stage_model(self, stage_num):
        """Save model for a specific stage."""
        save_dir = os.path.join(
            self.config['checkpoints_dir'],
            self.config['model'],
            self.config['test_no']
        )
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f'stage_{stage_num}_best.pth')
        torch.save(self.models[stage_num].state_dict(), save_path)
    
    def load_stage_model(self, stage_num):
        """Load model for a specific stage."""
        load_path = os.path.join(
            self.config['checkpoints_dir'],
            self.config['model'],
            self.config['test_no'],
            f'stage_{stage_num}_best.pth'
        )
        self.models[stage_num].load_state_dict(torch.load(load_path, weights_only=True))
    
    def save_cascade_image(self, img_idx, input_img, stage_outputs, target):
        """Save visualization showing progression through cascade."""

        num_plots = 2 + len(stage_outputs)  # input + stages + target
        
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        
        # Input
        axes[0].imshow(input_img.cpu().squeeze(), cmap='gray')
        axes[0].set_title('Noisy Input')
        axes[0].axis('off')
        
        # Stage outputs
        for i, output in enumerate(stage_outputs):
            psnr = compute_psnr(output, target)
            ssim = compute_ssim(output, target)
            
            axes[i+1].imshow(output.cpu().squeeze(), cmap='gray')
            axes[i+1].set_title(f'Stage {i}\nPSNR: {psnr:.2f}\nSSIM: {ssim:.4f}')
            axes[i+1].axis('off')
        
        # Target the 
        axes[-1].imshow(target.cpu().squeeze(), cmap='gray')
        axes[-1].set_title('Ground Truth')
        axes[-1].axis('off')
        
        plt.tight_layout()
        
        # Save the 
        save_dir = os.path.join(
            self.config['output_dir'],
            self.config['model'],
            'fig',
            self.config['test_no'],
            'cascade_results'
        )
        os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(os.path.join(save_dir, f'cascade_{img_idx:03d}.png'), dpi=150)
        plt.close()
    
    def plot_stage_loss_curves(self):
        """Plot loss curves for all stages."""
        fig, axes = plt.subplots(1, self.num_stages, figsize=(6 * self.num_stages, 5))
        
        if self.num_stages == 1: # convert the axes into list
            axes = [axes]
        
        for stage in range(self.num_stages):
            epochs = range(1, len(self.stage_train_losses[stage]) + 1)
            
            axes[stage].plot(epochs, self.stage_train_losses[stage], label='Train Loss', marker='o')
            axes[stage].plot(epochs, self.stage_val_losses[stage], label='Val Loss', marker='s')

            axes[stage].axvline(self.stage_best_epochs[stage] +1, linestyle='--', alpha=1, color='red')
            
            axes[stage].set_title(f'Stage {stage} Loss Curves')
            axes[stage].set_xlabel('Epoch')
            axes[stage].set_ylabel('Loss')
            axes[stage].legend()
            axes[stage].grid(True)
        
        plt.tight_layout()
        
        # Save
        save_dir = os.path.join(
            self.config['output_dir'],
            self.config['model'],
            'fig',
            self.config['test_no']
        )
        os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(os.path.join(save_dir, 'stage_loss_curves.png'), dpi=150)
        plt.close()
        
        self.logger.info(f"Loss curves saved to {save_dir}")

    def plot_stage_improvement(self):
        """Plot PSNR, SSIM, RMSE improvement across cascade stages."""

        baseline = self.test_results["baseline"]
        stages = self.test_results["stages"]

        psnr_values = [baseline["psnr"]] + stages["psnr"]
        ssim_values = [baseline["ssim"]] + stages["ssim"]
        rmse_values = [baseline["rmse"]] + stages["rmse"]

        stage_labels = ["Input"] + [f"S{i}" for i in range(self.num_stages)]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # PSNR
        axes[0].plot(stage_labels, psnr_values, marker='o')
        axes[0].set_title("PSNR Improvement Across Cascade")
        axes[0].set_xlabel("Stage")
        axes[0].set_ylabel("PSNR")
        axes[0].grid(True)

        # SSIM
        axes[1].plot(stage_labels, ssim_values, marker='o')
        axes[1].set_title("SSIM Improvement Across Cascade")
        axes[1].set_xlabel("Stage")
        axes[1].set_ylabel("SSIM")
        axes[1].grid(True)

        # RMSE
        axes[2].plot(stage_labels, rmse_values, marker='o')
        axes[2].set_title("RMSE Reduction Across Cascade")
        axes[2].set_xlabel("Stage")
        axes[2].set_ylabel("RMSE")
        axes[2].grid(True)

        plt.tight_layout()

        save_dir = os.path.join(
            self.config['output_dir'],
            self.config['model'],
            'fig',
            self.config['test_no']
        )

        os.makedirs(save_dir, exist_ok=True)

        plt.savefig(os.path.join(save_dir, "cascade_stage_improvement.png"), dpi=150)
        plt.close()

        self.logger.info(f"Stage improvement plot saved to {save_dir}")

    
    def run(self):
        """Run the complete training and testing pipeline."""
        self.train()
        self.test()