"""
Main experiment runner for Fisher Information Analysis.
Orchestrates the entire analysis pipeline from data loading to result generation.
"""

import torch
from datetime import datetime
from typing import Dict, Any, Tuple
import os
import json
from tqdm import tqdm
import numpy as np

from config import ExperimentConfig
from fisher_margin import FisherMargin
# from visualization import plot_fisher_margin_figures
from data_loader import load_data_and_model

class FisherExperimentRunner:
    """Main experiment runner for Fisher Information Analysis."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Load data and model
        print("Loading dataset and model...")
        self.test_ds, self.test_loader, self.transform, self.model = load_data_and_model(
            config.dataset_name, config.model_name
        )
        self.model.eval()
        
        # Get device from model
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on device: {self.device}")
        
        # Get number of classes from the dataset
        if hasattr(self.test_ds, 'classes'):
            self.num_classes = len(self.test_ds.classes)
            self.class_names = self.test_ds.classes
        else:
            # Fallback for datasets without class names
            self.num_classes = 10  # Default fallback
            self.class_names = [f'class_{i}' for i in range(self.num_classes)]
        
        # Initialize OSSA attack for Fisher analysis
        self.fisher_margin = FisherMargin(
            model=self.model,
            config=self.config
        )
        
        # Initialize results storage
        self.results = {}
    
    def _evaluate_model_accuracy(self, max_samples: int = None) -> Dict[str, Any]:
        """Evaluate the models accuracy on the test dataset."""
        print(f"Evaluating model accuracy on {self.config.dataset_name}...")
        
        # Model is already in eval mode from __init__, but ensure it's still eval
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.test_loader, desc="Evaluating accuracy")):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Get predictions
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Stop if max_samples is reached
                if max_samples is not None and total >= max_samples:
                    break
        
        accuracy = 100. * correct / total
        
        results = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'dataset_name': self.config.dataset_name,
            'model_name': self.config.model_name,
            'device': str(self.device)
        }
        
        print(f"✓ Model accuracy: {accuracy:0.2f} ({correct}/{total})")
        
        return results
    
    def _run_fisher_margin(self, x: torch.Tensor) -> Dict[str, Any]:
        """Run Fisher Margin Analysis."""
        
        import time
        
        # Start timing
        start_time = time.time()
        
        # Get initial prediction
        pred_label, confidence, probs = self.fisher_margin._get_model_prediction_and_probs(self.model, x)
        
        # Compute minimal geodesic margins
        margins, xk_tensor, deltas_tensor, alphas_list, probs_seq_list, pred_k_list, margin_losses_list, target_classes, num_steps_list = \
            self.fisher_margin.compute_min_geodesic_margin(
                model=self.model,
                x=x,
                cg_tol=self.config.cg_tol,
                cg_max_iter=self.config.cg_max_iter,
                max_steps=self.config.max_steps,
                lambda_reg=self.config.lambda_reg
            )
        
        # End timing
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Create visualizations
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Prepare results
        results = {
            'initial_prediction': {
                'class': pred_label,
                'class_name': self.class_names[pred_label],
                'confidence': confidence,
                'probabilities': probs.squeeze().cpu().tolist()
            },
            'margins': margins,
            'xk_tensor': xk_tensor,
            'deltas_tensor': deltas_tensor,
            'alphas_list': alphas_list,
            'probs_seq_list': probs_seq_list,
            'pred_k_list': pred_k_list,
            'margin_losses_list': margin_losses_list,
            'target_classes': target_classes,
            'num_steps_list': num_steps_list,
            'execution_time_seconds': execution_time,
            'analysis_parameters': {
                'cg_tol': self.config.cg_tol,
                'cg_max_iter': self.config.cg_max_iter,
                'max_steps': self.config.max_steps,
                'lambda_reg': self.config.lambda_reg,
                'sample_size': self.config.sample_size
            }
        }

        self._save_fisher_results(results)
        
        return results
    
    def _save_fisher_results(self, results: Dict[str, Any], sample_index: int = None) -> Dict[str, str]:
        """Save Fisher margin results as both JSON and tensor files.
        
        Args:
            results: The results dictionary from _run_fisher_margin
            sample_index: Optional index to include in filename for multiple samples
            
        Returns:
            Dict[str, str]: Dictionary containing paths to saved JSON and tensor files
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Extract class name from results if available
        class_name = None
        if 'initial_prediction' in results and 'class_name' in results['initial_prediction']:
            class_name = results['initial_prediction']['class_name']
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate base filename
        if sample_index is not None:
            if class_name:
                base_filename = f"fisher_results_sample_{sample_index}_{class_name}_{timestamp}"
            else:
                base_filename = f"fisher_results_sample_{sample_index}_{timestamp}"
        else:
            if class_name:
                base_filename = f"fisher_results_{class_name}_{timestamp}"
            else:
                base_filename = f"fisher_results_{timestamp}"
        
        # Save JSON results (excluding tensors for JSON)
        json_results = results.copy()
        # Remove tensor data from JSON to avoid serialization issues
        json_results.pop('xk_tensor', None)
        json_results.pop('deltas_tensor', None)
        
        json_filepath = os.path.join(self.config.output_dir, f"{base_filename}.json")
        with open(json_filepath, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save tensor data
        # tensor_data = {}
        
        # # Save xk_tensor if available
        # if 'xk_tensor' in results and results['xk_tensor'] is not None:
        #     tensor_data['xk_tensor'] = results['xk_tensor']
        
        # # Save deltas_tensor if available
        # if 'deltas_tensor' in results and results['deltas_tensor'] is not None:
        #     tensor_data['deltas_tensor'] = results['deltas_tensor']
        
        # # Save additional metadata
        # tensor_data['metadata'] = {
        #     'class_name': class_name,
        #     'timestamp': timestamp,
        #     'sample_index': sample_index,
        #     'tensor_shapes': {
        #         'xk_tensor': results.get('xk_tensor', None).shape if results.get('xk_tensor') is not None else None,
        #         'deltas_tensor': results.get('deltas_tensor', None).shape if results.get('deltas_tensor') is not None else None
        #     }
        # }
        
        # tensor_filepath = os.path.join(self.config.output_dir, f"tensor_{base_filename}.pt")
        # torch.save(tensor_data, tensor_filepath)