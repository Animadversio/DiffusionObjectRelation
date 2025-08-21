# Generalization Profile Evaluation CLI

This CLI script evaluates model generalization across different prompt templates and training checkpoints. It's based on the notebook `20250819_DiT_T5_generalization_profile.ipynb` and optimized for efficiency through in-memory prompt embedding caching.

## Key Features

- **Pre-computes embeddings once**: Text embeddings are computed once and reused across all checkpoints
- **Multiple prompt templates**: Test different linguistic formulations of spatial relationships
- **Comprehensive evaluation**: Evaluates shape, color, spatial relationship accuracy, and position metrics
- **Batch processing**: Evaluates multiple checkpoints automatically
- **Structured output**: Results saved in CSV format with detailed metrics

## Usage

### Basic Usage

```bash
# Evaluate all checkpoints for a T5-based model
python generalization_profile_eval_cli.py --model_run_name objrel_T5_DiT_B_pilot

# Evaluate specific checkpoints only
python generalization_profile_eval_cli.py \
    --model_run_name objrel_T5_DiT_B_pilot \
    --checkpoints epoch_1500_step_60000.pth epoch_4000_step_160000.pth

# Use single prompt mode (22 prompts instead of 264)
python generalization_profile_eval_cli.py \
    --model_run_name objrel_T5_DiT_B_pilot \
    --single_prompt_mode

# Custom prompt templates
python generalization_profile_eval_cli.py \
    --model_run_name objrel_T5_DiT_B_pilot \
    --prompt_templates "{color1} {shape1} is {rel_text} {color2} {shape2}" \
                      "the {color1} {shape1} {rel_text} the {color2} {shape2}"
```

### Advanced Usage

```bash
# Full customization
python generalization_profile_eval_cli.py \
    --model_run_name objrel_rndembdposemb_DiT_B_pilot \
    --text_encoder_type RandomEmbeddingEncoder_wPosEmb \
    --num_images 25 \
    --num_inference_steps 20 \
    --guidance_scale 5.0 \
    --output_dir /custom/output/path \
    --generator_seed 123
```

## Arguments

### Required
- `--model_run_name`: Model to evaluate (from predefined list)

### Optional
- `--checkpoints`: Specific checkpoint files (default: all available)
- `--text_encoder_type`: Text encoder type (auto-determined if not specified)
- `--prompt_templates`: Prompt templates to test (default: 3 templates)
- `--num_images`: Images per prompt (default: 49)
- `--num_inference_steps`: Inference steps (default: 14) 
- `--guidance_scale`: Guidance scale (default: 4.5)
- `--max_sequence_length`: Max text sequence length (default: 20)
- `--generator_seed`: Random seed (default: 42)
- `--output_dir`: Custom output directory
- `--single_prompt_mode`: Use 22 prompts instead of 264

## Output Structure

```
results/{model_name}/generalization_eval/
├── eval_df_{checkpoint}_{template}.csv           # Per-checkpoint per-template results
├── eval_df_{checkpoint}_all_templates.csv       # Per-checkpoint combined results  
├── eval_df_all_checkpoints_all_templates.csv    # All results combined
└── summary_across_checkpoints_templates.csv     # Summary statistics
```

## Supported Models

- `objrel_T5_DiT_B_pilot` (T5 encoder)
- `objrel_T5_DiT_mini_pilot` (T5 encoder)
- `objrel_rndembdposemb_DiT_B_pilot` (Random embedding + positional)
- `objrel_rndembdposemb_DiT_micro_pilot` (Random embedding + positional)
- `objrel_rndembdposemb_DiT_nano_pilot` (Random embedding + positional)
- `objrel_rndembdposemb_DiT_mini_pilot` (Random embedding + positional)
- `objrel_rndemb_DiT_B_pilot` (Random embedding only)
- `objrel_T5_DiT_B_pilot_WDecay` (T5 encoder with weight decay)
- `objrel_T5_DiT_mini_pilot_WDecay` (T5 encoder with weight decay)

## Default Prompt Templates

1. `"{color1} {shape1} is {rel_text} {color2} {shape2}"` 
2. `"{color1} {shape1} {rel_text} the {color2} {shape2}"`
3. `"the {color1} {shape1} {rel_text} the {color2} {shape2}"`

## Performance Optimizations

1. **In-Memory Embedding Cache**: Embeddings computed once, stored in CPU memory, moved to GPU only during inference
2. **Batch Checkpoint Processing**: Model weights updated without recomputing text embeddings
3. **GPU Memory Management**: Automatic cache clearing between evaluations
4. **Progress Tracking**: Detailed progress bars for long-running evaluations

## Evaluation Metrics

- **overall**: Combined accuracy across all aspects
- **shape**: Shape recognition accuracy
- **color**: Color recognition accuracy
- **spatial_relationship**: Exact spatial relationship accuracy
- **spatial_relationship_loose**: Relaxed spatial relationship accuracy
- **Dx, Dy**: Position displacement metrics
- **exist_binding, unique_binding**: Object binding metrics

## Example Output

```
Evaluating model: objrel_T5_DiT_B_pilot
Text encoder type: T5
Prompt templates: 3
Single prompt mode: False

Step 1: Loading text encoder...
Step 2: Generating prompt collections...
  Template '{color1} {shape1} is {rel_text} {color2} {shape2}': 264 prompts
  Template '{color1} {shape1} {rel_text} the {color2} {shape2}': 264 prompts
  Template 'the {color1} {shape1} {rel_text} the {color2} {shape2}': 264 prompts

Step 3: Pre-computing embeddings...
Computing embeddings: 100%|██████| 792/792 [00:15<00:00, 52.1it/s]
Cached embeddings for 792 unique prompts

Step 4: Setting up pipeline...
Step 5: Found 2 checkpoints to evaluate

Evaluating checkpoint: epoch_1500_step_60000.pth
  Checkpoint epoch_1500_step_60000.pth summary: {'overall': 0.734, 'shape': 0.944, 'color': 0.968, 'spatial_relationship': 0.734}

Evaluating checkpoint: epoch_4000_step_160000.pth  
  Checkpoint epoch_4000_step_160000.pth summary: {'overall': 0.894, 'shape': 0.951, 'color': 0.948, 'spatial_relationship': 0.894}

Step 7: Generating summary...
Evaluation complete! Results saved to: results/objrel_T5_DiT_B_pilot/generalization_eval
Total samples evaluated: 77616
```