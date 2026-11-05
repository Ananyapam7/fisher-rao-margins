"""
Configuration module for Fisher Information Analysis experiments.
Centralizes all experiment settings, hyperparameters, and model configurations.
"""

import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os

@dataclass
class ExperimentConfig:
    """Configuration for Fisher Information Analysis experiments."""
    
    # Dataset and Model Configuration
    dataset_name: str = 'cifar10'   # Options: 'cifar10'
    model_name: str = 'resnet18'    # Options: 'resnet18', 'resnet50', 'vgg11_bn', 'densenet121', etc.
    
    # Output Configuration
    output_dir: str = 'experiments'
    plot_format: str = 'png'        # Format for saved plots ('png', 'pdf', 'svg')
    plot_dpi: int = 300            # DPI for saved plots
    
    sample_size: int = 20            # Number of Monte Carlo samples per iteration
    
    # CG Solver Configuration
    cg_tol: float = 1e-3            # Tolerance for CG solves
    cg_max_iter: int = 30           # Maximum CG iterations
    max_steps: int = 100            # Maximum steps per class
    lambda_reg: float = 0.01        # Regularization parameter
    
    # Device Configuration
    device: Optional[str] = None    # If None, will auto-detect
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    @property
    def dataset_normalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get normalization parameters for the selected dataset."""
        if self.dataset_name == 'cifar10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2471, 0.2435, 0.2616]).view(3, 1, 1)
        else:
            # Default to CIFAR-10 normalization
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2471, 0.2435, 0.2616]).view(3, 1, 1)
        
        return mean, std
    
    def get_output_path(self, filename: str) -> str:
        """Get full output path for a file."""
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.normpath(os.path.join(self.output_dir, filename))
    
    def print_config(self):
        """Print current configuration."""
        print("=" * 60)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Output Directory: {self.output_dir}")
        print(f"  Sample Size: {self.sample_size}")
        print(f"CG Solver Parameters:")
        print(f"  CG Tolerance: {self.cg_tol}")
        print(f"  CG Max Iterations: {self.cg_max_iter}")
        print(f"  Max Steps: {self.max_steps}")
        print(f"  Lambda Regularization: {self.lambda_reg}")
        print("=" * 60) 