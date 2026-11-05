# Experiment 2: Model Accuracy Evaluation on CIFAR-10

## Overview
This experiment evaluates the accuracy of all available pre-trained models on the CIFAR-10 test dataset. It provides a comprehensive comparison of different model architectures and their performance.

## Available Models
The experiment evaluates the following 13 models:
- **ResNet Family**: resnet18, resnet34, resnet50
- **VGG Family**: vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn  
- **DenseNet Family**: densenet121, densenet161, densenet169
- **Other Architectures**: googlenet, inception_v3, mobilenet_v2

## Configuration
- **Dataset**: CIFAR-10 (10,000 test samples)
- **Evaluation**: Full test set accuracy
- **Device**: CPU/GPU (auto-detected)
- **Precision**: Double precision for numerical stability

## Purpose
- Evaluate and compare accuracy of all available models
- Identify the best performing model on CIFAR-10
- Provide baseline accuracy measurements for further analysis
- Generate comprehensive accuracy ranking

## Output
- **JSON Results**: Detailed accuracy results for each model
- **Text Summary**: Human-readable accuracy ranking and statistics
- **Console Output**: Real-time progress and final ranking

## Usage
```bash
cd experiments/experiment2
python experiment_runner.py
```

## Expected Runtime
~10-30 minutes (depending on hardware and number of successful model loads)

## Output Files
- `model_accuracy_results_YYYYMMDD_HHMMSS.json`: Complete results in JSON format
- `accuracy_summary_YYYYMMDD_HHMMSS.txt`: Human-readable summary
- Console output with real-time progress and final ranking

## Key Features
- **Comprehensive Evaluation**: Tests all 13 available models
- **Error Handling**: Gracefully handles models that fail to load
- **Ranking System**: Automatically ranks models by accuracy
- **Statistics**: Provides mean, best, and worst accuracy statistics
- **Detailed Logging**: Tracks successful and failed evaluations

## Expected Results
- Accuracy rankings from best to worst performing models
- Statistical summary of all successful evaluations
- List of any models that failed to load or evaluate
- Complete results saved for further analysis 