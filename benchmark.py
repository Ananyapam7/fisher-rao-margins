"""
Benchmarking module for Fisher Information Analysis.
Supports ablation studies, baseline comparisons, and systematic experiments.
"""

import torch
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from config import ExperimentConfig
from experiment_runner import FisherExperimentRunner

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    config_name: str
    dataset_name: str
    model_name: str
    execution_time: float
    memory_usage_mb: float
    top_eigenvalue: float
    timestamp: str

class FisherBenchmarker:
    """Benchmarking class for Fisher Information Analysis."""
    
    def __init__(self, base_config: ExperimentConfig):
        self.base_config = base_config
        self.results: List[BenchmarkResult] = []
        
    def run_ablation_study(self, ablation_configs: List[Dict[str, Any]]) -> List[BenchmarkResult]:
        """
        Run ablation study with different configurations.
        
        Args:
            ablation_configs: List of configuration dictionaries to test
            
        Returns:
            List of benchmark results
        """
        print("="*60)
        print("STARTING ABLATION STUDY")
        print("="*60)
        
        results = []
        
        for i, config_dict in enumerate(ablation_configs):
            print(f"\n--- Ablation {i+1}/{len(ablation_configs)} ---")
            
            # Create config from base config
            config = ExperimentConfig(**{**asdict(self.base_config), **config_dict})
            config.output_dir = os.path.normpath(f"{self.base_config.output_dir}/ablation_{i+1}")
            
            # Run experiment
            result = self._run_single_benchmark(config, f"ablation_{i+1}")
            results.append(result)
            
        self.results.extend(results)
        return results
    
    def run_model_comparison(self, models: List[str], dataset: str = None) -> List[BenchmarkResult]:
        """
        Compare different models on the same dataset.
        
        Args:
            models: List of model names to compare
            dataset: Dataset to use (uses base config if None)
            
        Returns:
            List of benchmark results
        """
        print("="*60)
        print("STARTING MODEL COMPARISON")
        print("="*60)
        
        results = []
        
        for i, model_name in enumerate(models):
            print(f"\n--- Model {i+1}/{len(models)}: {model_name} ---")
            
            # Create config
            config = ExperimentConfig(**asdict(self.base_config))
            config.model_name = model_name
            if dataset:
                config.dataset_name = dataset
            config.output_dir = os.path.normpath(f"{self.base_config.output_dir}/model_{model_name}")
            
            # Run experiment
            result = self._run_single_benchmark(config, f"model_{model_name}")
            results.append(result)
            
        self.results.extend(results)
        return results
    
    def run_dataset_comparison(self, datasets: List[str], model: str = None) -> List[BenchmarkResult]:
        """
        Compare different datasets with the same model.
        
        Args:
            datasets: List of dataset names to compare
            model: Model to use (uses base config if None)
            
        Returns:
            List of benchmark results
        """
        print("="*60)
        print("STARTING DATASET COMPARISON")
        print("="*60)
        
        results = []
        
        for i, dataset_name in enumerate(datasets):
            print(f"\n--- Dataset {i+1}/{len(datasets)}: {dataset_name} ---")
            
            # Create config
            config = ExperimentConfig(**asdict(self.base_config))
            config.dataset_name = dataset_name
            if model:
                config.model_name = model
            config.output_dir = os.path.normpath(f"{self.base_config.output_dir}/dataset_{dataset_name}")
            
            # Run experiment
            result = self._run_single_benchmark(config, f"dataset_{dataset_name}")
            results.append(result)
            
        self.results.extend(results)
        return results
    
    def run_hyperparameter_sweep(self, param_name: str, param_values: List[Any]) -> List[BenchmarkResult]:
        """
        Sweep over a single hyperparameter.
        
        Args:
            param_name: Name of the parameter to sweep
            param_values: List of values to test
            
        Returns:
            List of benchmark results
        """
        print("="*60)
        print(f"STARTING HYPERPARAMETER SWEEP: {param_name}")
        print("="*60)
        
        results = []
        
        for i, value in enumerate(param_values):
            print(f"\n--- {param_name} = {value} ({i+1}/{len(param_values)}) ---")
            
            # Create config
            config = ExperimentConfig(**asdict(self.base_config))
            setattr(config, param_name, value)
            config.output_dir = os.path.normpath(f"{self.base_config.output_dir}/sweep_{param_name}_{value}")
            
            # Run experiment
            result = self._run_single_benchmark(config, f"sweep_{param_name}_{value}")
            results.append(result)
            
        self.results.extend(results)
        return results
    
    def _run_single_benchmark(self, config: ExperimentConfig, config_name: str) -> BenchmarkResult:
        """Run a single benchmark experiment."""
        start_time = time.time()
        
        # Get initial memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        else:
            initial_memory = 0
        
        try:
            # Run experiment
            runner = FisherExperimentRunner(config)
            results = runner.run_full_experiment()
            
            # Extract key metrics
            # Use the first fisher analysis result (fisher_analysis_0)
            fisher_results = results.get('fisher_analysis_0', {})
            
            execution_time = time.time() - start_time
            
            # Get peak memory usage
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                memory_usage = peak_memory - initial_memory
            else:
                memory_usage = 0
            
            # Create benchmark result
            benchmark_result = BenchmarkResult(
                config_name=config_name,
                dataset_name=config.dataset_name,
                model_name=config.model_name,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                top_eigenvalue=fisher_results.get('min_margin', 0.0),  # Use min_margin as top eigenvalue proxy
                timestamp=datetime.now().isoformat()
            )
            
            print(f"✓ Completed in {execution_time:.2f}s, Memory: {memory_usage:.1f}MB")
            return benchmark_result
            
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            # Return a failed result
            return BenchmarkResult(
                config_name=config_name,
                dataset_name=config.dataset_name,
                model_name=config.model_name,
                execution_time=time.time() - start_time,
                memory_usage_mb=0,
                top_eigenvalue=0.0,
                timestamp=datetime.now().isoformat()
            )
    
    def save_results(self, filename: str = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        # Convert results to dictionaries
        results_dict = [asdict(result) for result in self.results]
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Saved benchmark results to: {filename}")
    
    def print_summary(self):
        """Print a summary of all benchmark results."""
        if not self.results:
            print("No benchmark results available.")
            return
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Group by experiment type
        experiment_types = {}
        for result in self.results:
            exp_type = result.config_name.split('_')[0]
            if exp_type not in experiment_types:
                experiment_types[exp_type] = []
            experiment_types[exp_type].append(result)
        
        for exp_type, results in experiment_types.items():
            print(f"\n{exp_type.upper()} EXPERIMENTS:")
            print("-" * 40)
            
            for result in results:
                print(f"  {result.config_name}:")
                print(f"    Dataset: {result.dataset_name}, Model: {result.model_name}")
                print(f"    Time: {result.execution_time:.2f}s, Memory: {result.memory_usage_mb:.1f}MB")
                print(f"    Top λ: {result.top_eigenvalue:.4f}")
                print()
    
    def get_best_config(self, metric: str = 'top_eigenvalue', maximize: bool = True) -> Optional[BenchmarkResult]:
        """Get the best configuration based on a metric."""
        if not self.results:
            return None
        
        if maximize:
            best_result = max(self.results, key=lambda x: getattr(x, metric))
        else:
            best_result = min(self.results, key=lambda x: getattr(x, metric))
        
        return best_result

# Predefined ablation configurations
ABLATION_CONFIGS = [
    # Baseline
    {},
    
    {'eps': 0.01},
    {'eps': 0.05},
    {'num_iters': 10},
    {'num_iters': 20},
    {'sample_size': 3},
    {'sample_size': 10},
    
    # Different CG solver parameters
    {'cg_tol': 1e-4},
    {'cg_tol': 1e-2},
    {'cg_max_iter': 20},
    {'cg_max_iter': 50},
    {'lambda_reg': 0.001},
    {'lambda_reg': 0.1},
]

# Predefined model comparisons
MODEL_COMPARISONS = [
    ['resnet18', 'resnet50'],
]

# Predefined dataset comparisons (only CIFAR-10 supported now)
DATASET_COMPARISONS = [
    ['cifar10'],
] 