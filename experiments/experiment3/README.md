# Experiment 3: Model Comparison Study

## Overview
This experiment compares Fisher Information Analysis across different model architectures on CIFAR-10. It analyzes how different model architectures affect the Fisher margins and provides insights into model robustness.

## Configuration
- **Dataset**: CIFAR-10
- **Models**: ResNet18, ResNet50, VGG11_BN
- **Max Steps**: 15
- **CG Max Iterations**: 25
- **CG Tolerance**: 1e-3
- **Lambda Regularization**: 0.01
- **Sample Size**: 15 (Monte Carlo samples)
- **Number of Test Images**: 5 per model

## Purpose
- Compare Fisher margins across different model architectures
- Analyze how model complexity affects robustness
- Identify which architectures are more robust to adversarial perturbations
- Provide insights for model selection based on Fisher information

## Output
- Fisher margin analysis results for each model
- Individual visualization plots for each model
- Model comparison summary with statistics
- JSON results files for each model
- Overall comparison summary in JSON format

## Usage
```bash
cd experiments/experiment3
python experiment_runner.py
```

## Expected Runtime
~45-90 minutes (depending on hardware)

## Models Tested
1. **ResNet18**: Lightweight residual network
2. **ResNet50**: Deeper residual network
3. **VGG11_BN**: VGG-style network with batch normalization

## Key Metrics
- Mean Fisher margin per model
- Standard deviation of margins
- Minimum and maximum margins
- Statistical comparison between models

## Error Handling
The experiment includes robust error handling to continue testing other models even if one fails to load or run properly. 