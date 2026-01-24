# Model Comparison Guide

This guide explains how to systematically evaluate and compare all 9 model variants in your object relationship study.

## Model Variants Overview

You have 9 different models with the following characteristics:

### Text Encoders (3 types):
- **T5**: Natural language understanding
- **Random**: Random embeddings without position information  
- **Random+Position**: Random embeddings with position encoding

### Model Sizes (4 types):
- **Base**: Largest model
- **Mini**: Medium model
- **Micro**: Small model
- **Nano**: Smallest model

### Training Strategies (2 types):
- **Standard**: Regular training
- **With Weight Decay**: Training with weight decay regularization

## Model Configurations

| Model Name | Text Encoder | Model Size | Training Strategy |
|------------|--------------|------------|-------------------|
| `objrel_T5_DiT_B_pilot` | T5 | Base | Standard |
| `objrel_T5_DiT_mini_pilot` | T5 | Mini | Standard |
| `objrel_rndembdposemb_DiT_B_pilot` | Random+Position | Base | Standard |
| `objrel_rndembdposemb_DiT_micro_pilot` | Random+Position | Micro | Standard |
| `objrel_rndembdposemb_DiT_nano_pilot` | Random+Position | Nano | Standard |
| `objrel_rndembdposemb_DiT_mini_pilot` | Random+Position | Mini | Standard |
| `objrel_rndemb_DiT_B_pilot` | Random | Base | Standard |
| `objrel_T5_DiT_B_pilot_WDecay` | T5 | Base | With Weight Decay |
| `objrel_T5_DiT_mini_pilot_WDecay` | T5 | Mini | With Weight Decay |

## Step-by-Step Comparison Process

### Step 1: Run Individual Model Evaluations

You can run evaluations for individual models using:

```bash
# Run a single model evaluation
python experimental_scripts/run_single_model_eval.py objrel_T5_DiT_B_pilot

# Or specify the text encoder explicitly
python experimental_scripts/run_single_model_eval.py objrel_rndemb_DiT_B_pilot RandomEmbeddingEncoder
```

### Step 2: Run All Models (Batch Processing)

To evaluate all 9 models systematically:

```bash
python experimental_scripts/batch_model_evaluation.py
```

This script will:
- Run evaluations for all models
- Collect results from saved files
- Create comparison plots
- Generate summary statistics

### Step 3: Detailed Analysis

For comprehensive analysis of all models:

```bash
python experimental_scripts/model_comparison_analysis.py
```

This will provide:
- Performance rankings
- Statistical significance tests
- Detailed visualizations
- Summary reports

## Expected Outputs

### 1. Individual Model Results
Each model evaluation creates:
- `eval_df_all_train_traj_prompts.csv`: Detailed evaluation metrics
- `object_df_all_train_traj_prompts.pkl`: Object detection data
- Training trajectory plots

### 2. Comparison Analysis
The analysis scripts create:
- `Figures/model_comparison/`: Comparison plots and heatmaps
- `Figures/model_analysis/`: Detailed analysis results
- Summary statistics and reports

## Key Research Questions

This setup allows you to answer:

1. **Text Encoder Effect**: How do T5 vs Random vs Random+Position encoders compare?
2. **Model Size Effect**: How does performance scale with model size?
3. **Training Strategy Effect**: Does weight decay improve performance?
4. **Interaction Effects**: Which combinations work best?

## Analysis Metrics

Each model is evaluated on:
- **Overall Accuracy**: Combined performance across all metrics
- **Color Accuracy**: Correct object colors
- **Shape Accuracy**: Correct object shapes  
- **Spatial Relationship Accuracy**: Correct positioning
- **Object Binding**: Object detection and association
- **Positional Metrics**: Dx, Dy distance measurements

## Running the Analysis

### Quick Start (if you have existing results):
```bash
# Just analyze existing results
python experimental_scripts/model_comparison_analysis.py
```

### Full Pipeline (run all evaluations):
```bash
# Run all evaluations first
python experimental_scripts/batch_model_evaluation.py

# Then analyze
python experimental_scripts/model_comparison_analysis.py
```

## Expected Timeline

- **Single model evaluation**: ~1-2 hours per model
- **All 9 models**: ~9-18 hours total
- **Analysis**: ~10-30 minutes

## Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce batch size or use smaller models first
2. **Timeout**: Increase timeout in scripts or run models individually
3. **Missing Checkpoints**: Ensure all model checkpoints exist before running

### Debugging:
```bash
# Test single model first
python experimental_scripts/run_single_model_eval.py objrel_T5_DiT_mini_pilot

# Check if results exist
ls /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/analysis_results/*/large_scale_eval_posthoc/
```

## Output Interpretation

The analysis will show you:

1. **Best Performing Model**: Which configuration achieves highest accuracy
2. **Encoder Comparison**: T5 vs Random vs Random+Position performance
3. **Size Scaling**: How performance scales with model size
4. **Training Effects**: Impact of weight decay
5. **Statistical Significance**: Which differences are meaningful

This systematic approach will give you comprehensive insights into how different architectural choices affect object relationship understanding in diffusion models. 