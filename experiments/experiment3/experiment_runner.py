#!/usr/bin/env python3
"""
Experiment 3: Model Comparison Study
Compare Fisher Information Analysis across different model architectures.
"""

import os
import sys
import torch
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import ExperimentConfig
from experiment_runner import FisherExperimentRunner

def run_experiment3():
    """Run Experiment 3: Model Comparison Study."""
    print("=" * 60)
    print("EXPERIMENT 3: Model Comparison Study")
    print("Comparing Fisher Information Analysis across model architectures")
    print("=" * 60)
    
    # Define models to compare
    models_to_test = ['resnet18', 'resnet50', 'vgg11_bn']
    
    # Store results for all models
    all_results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*40}")
        print(f"Testing Model: {model_name}")
        print(f"{'='*40}")
        
        # Create configuration for this model
        config = ExperimentConfig(
            dataset_name='cifar10',
            model_name=model_name,
            output_dir=f'experiments/experiment3/results/{model_name}',
            max_steps=15,
            cg_max_iter=25,
            cg_tol=1e-3,
            lambda_reg=0.01,
            sample_size=15
        )
        
        # Print configuration
        config.print_config()
        
        try:
            # Initialize the experiment runner
            print(f"\nInitializing experiment runner for {model_name}...")
            runner = FisherExperimentRunner(config)

            # Run Fisher margin for 5 images from the test set
            print(f"Running Fisher margin analysis for 5 test images with {model_name}...")
            idx = 0
            max_images = 5
            model_results = {}
            
            for batch in tqdm(runner.test_loader, desc=f"Fisher margin ({model_name})"):
                data, target = batch
                data = data.to(runner.device)
                # If batch size > 1, iterate over each image in the batch
                for i in range(data.size(0)):
                    if idx >= max_images:
                        break
                    x = data[i].to(torch.float64)
                    result = runner._run_fisher_margin(x.unsqueeze(0))
                    model_results[f'fisher_analysis_{idx}'] = result
                    idx += 1
                if idx >= max_images:
                    break
            
            # Save individual model results
            runner.results = model_results
            runner._save_fisher_results_json(runner.results)
            
            # Calculate summary statistics for this model
            margins_all = []
            for key, result in model_results.items():
                if 'fisher_analysis' in key:
                    margins = result.get('margins', [])
                    margins_all.extend(margins)
            
            if margins_all:
                margins_tensor = torch.tensor(margins_all)
                model_summary = {
                    'model_name': model_name,
                    'total_margins': len(margins_all),
                    'mean_margin': margins_tensor.mean().item(),
                    'std_margin': margins_tensor.std().item(),
                    'min_margin': margins_tensor.min().item(),
                    'max_margin': margins_tensor.max().item(),
                    'timestamp': datetime.now().isoformat()
                }
                all_results[model_name] = model_summary
                
                print(f"\nSummary for {model_name}:")
                print(f"  Total margins: {len(margins_all)}")
                print(f"  Mean margin: {model_summary['mean_margin']:.4f}")
                print(f"  Std margin: {model_summary['std_margin']:.4f}")
                print(f"  Min margin: {model_summary['min_margin']:.4f}")
                print(f"  Max margin: {model_summary['max_margin']:.4f}")
            
            # Check if output files were created
            output_files = os.listdir(config.output_dir)
            print(f"  Output files created: {len(output_files)}")
            
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
            all_results[model_name] = {
                'model_name': model_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Save comparison summary
    comparison_file = 'experiments/experiment3/results/model_comparison_summary.json'
    os.makedirs(os.path.dirname(comparison_file), exist_ok=True)
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print final comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    for model_name, summary in all_results.items():
        if 'error' not in summary:
            print(f"\n{model_name}:")
            print(f"  Mean margin: {summary['mean_margin']:.4f}")
            print(f"  Std margin: {summary['std_margin']:.4f}")
            print(f"  Min margin: {summary['min_margin']:.4f}")
            print(f"  Max margin: {summary['max_margin']:.4f}")
        else:
            print(f"\n{model_name}: ERROR - {summary['error']}")
    
    print("\n" + "=" * 60)
    print("✓ EXPERIMENT 3 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    run_experiment3() 