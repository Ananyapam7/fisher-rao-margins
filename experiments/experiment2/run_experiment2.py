#!/usr/bin/env python3
"""
Experiment 2: Model Accuracy Evaluation on CIFAR-10
Evaluate the accuracy of all available models on the CIFAR-10 test dataset.
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

def run_experiment2():
    """Run Experiment 2: Model Accuracy Evaluation on CIFAR-10."""
    print("=" * 60)
    print("EXPERIMENT 2: Model Accuracy Evaluation on CIFAR-10")
    print("Evaluating all available models on CIFAR-10 test dataset")
    print("=" * 60)
    
    # Define all available models for CIFAR-10
    available_models = [
        'resnet18',
        'resnet34', 
        'resnet50',
        'vgg11_bn',
        'vgg13_bn',
        'vgg16_bn',
        'vgg19_bn',
        'densenet121',
        'densenet161',
        'densenet169',
        'googlenet',
        'inception_v3',
        'mobilenet_v2'
    ]
    
    # Create output directory
    output_dir = 'experiments/experiment2/results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all accuracy results
    all_results = {
        'experiment_info': {
            'name': 'Model Accuracy Evaluation on CIFAR-10',
            'timestamp': datetime.now().isoformat(),
            'dataset': 'cifar10',
            'test_samples': 10000,  # Full CIFAR-10 test set
            'models_evaluated': len(available_models)
        },
        'model_accuracies': {}
    }
    
    print(f"\nEvaluating {len(available_models)} models on CIFAR-10 test dataset...")
    print("=" * 60)
    
    # Evaluate each model
    for i, model_name in enumerate(available_models, 1):
        print(f"\n[{i}/{len(available_models)}] Evaluating {model_name}...")
        print("-" * 40)
        
        try:
            # Create a basic configuration for this model
            config = ExperimentConfig(
                dataset_name='cifar10',
                model_name=model_name,
                output_dir=output_dir,
                max_steps=10,  # Not used for accuracy evaluation
                cg_max_iter=10,
                cg_tol=1e-3,
                lambda_reg=0.001,
                sample_size=10
            )
            
            # Initialize the experiment runner
            runner = FisherExperimentRunner(config)
            
            # Evaluate model accuracy on full test set
            accuracy_results = runner._evaluate_model_accuracy(max_samples=None)
            
            # Store results
            all_results['model_accuracies'][model_name] = accuracy_results
            
            print(f"✓ {model_name}: {accuracy_results['accuracy']:.2f}% accuracy")
            
        except Exception as e:
            print(f"✗ Error evaluating {model_name}: {str(e)}")
            all_results['model_accuracies'][model_name] = {
                'error': str(e),
                'accuracy': None,
                'correct': None,
                'total': None
            }
    
    # Generate summary statistics
    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    
    successful_models = []
    failed_models = []
    
    for model_name, results in all_results['model_accuracies'].items():
        if 'error' not in results:
            successful_models.append((model_name, results['accuracy']))
            print(f"{model_name:15s}: {results['accuracy']:6.2f}%")
        else:
            failed_models.append(model_name)
            print(f"{model_name:15s}: FAILED - {results['error']}")
    
    if successful_models:
        # Sort by accuracy (descending)
        successful_models.sort(key=lambda x: x[1], reverse=True)
        
        print("\n" + "-" * 40)
        print("RANKING BY ACCURACY (Best to Worst)")
        print("-" * 40)
        for i, (model_name, accuracy) in enumerate(successful_models, 1):
            print(f"{i:2d}. {model_name:15s}: {accuracy:6.2f}%")
        
        # Calculate statistics
        accuracies = [acc for _, acc in successful_models]
        mean_acc = sum(accuracies) / len(accuracies)
        max_acc = max(accuracies)
        min_acc = min(accuracies)
        
        print(f"\nStatistics:")
        print(f"  Mean accuracy: {mean_acc:.2f}%")
        print(f"  Best accuracy: {max_acc:.2f}% ({successful_models[0][0]})")
        print(f"  Worst accuracy: {min_acc:.2f}% ({successful_models[-1][0]})")
        print(f"  Successful evaluations: {len(successful_models)}/{len(available_models)}")
    
    if failed_models:
        print(f"\nFailed evaluations: {len(failed_models)}")
        for model in failed_models:
            print(f"  - {model}")
    
    # Save results to JSON
    results_file = os.path.join(output_dir, f'model_accuracy_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create a simple text summary
    summary_file = os.path.join(output_dir, f'accuracy_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(summary_file, 'w') as f:
        f.write("CIFAR-10 Model Accuracy Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Models evaluated: {len(available_models)}\n")
        f.write(f"Successful evaluations: {len(successful_models)}\n")
        f.write(f"Failed evaluations: {len(failed_models)}\n\n")
        
        if successful_models:
            f.write("ACCURACY RANKING:\n")
            f.write("-" * 20 + "\n")
            for i, (model_name, accuracy) in enumerate(successful_models, 1):
                f.write(f"{i:2d}. {model_name:15s}: {accuracy:6.2f}%\n")
            
            f.write(f"\nSTATISTICS:\n")
            f.write(f"Mean accuracy: {mean_acc:.2f}%\n")
            f.write(f"Best accuracy: {max_acc:.2f}% ({successful_models[0][0]})\n")
            f.write(f"Worst accuracy: {min_acc:.2f}% ({successful_models[-1][0]})\n")
        
        if failed_models:
            f.write(f"\nFAILED MODELS:\n")
            for model in failed_models:
                f.write(f"- {model}\n")
    
    print(f"Summary saved to: {summary_file}")
    
    # Check if output files were created
    output_files = os.listdir(output_dir)
    print(f"\nOutput files created: {len(output_files)}")
    for file in output_files:
        print(f"  - {file}")
    
    print("\n" + "=" * 60)
    print("✓ EXPERIMENT 2 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    run_experiment2() 