"""
data_generation_testing.py

Comprehensive testing script for the unified dataset interface.
Includes visual inspection, statistical analysis, and validation of dataset outputs.

Authors: Hannah Kim
Date: 2025-07-29
"""

import os
import sys
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageDraw

# Add paths for imports
sys.path.append("/n/home12/hjkim/Github/DiffusionObjectRelation/PixArt-alpha")
sys.path.append("/n/home12/hjkim/Github/DiffusionObjectRelation/")

from utils.dataset_generation_lib import SingleShapeDataset, DoubleShapeDataset, MixedShapesDataset

def create_dataset(dataset_type, num_images=1000, resolution=64, radius=16, **kwargs):
    """
    Create any type of dataset with unified parameters.
    
    Parameters:
    - dataset_type: str, one of "Single", "Double", "Mixed"
    - num_images: int, number of images to generate
    - resolution: int, image resolution (width=height)
    - radius: int, radius/size of shapes
    - **kwargs: additional parameters (e.g., single_ratio for Mixed dataset)
    
    Returns:
    - dataset: Dataset object
    """
    
    # Define transform
    transform = transforms.Compose([
        lambda x: x.convert("RGB"),
        transforms.ToTensor(),
    ])
    
    # Dataset class mapping
    dataset_classes = {
        "Single": SingleShapeDataset,
        "Double": DoubleShapeDataset,
        "Mixed": MixedShapesDataset
    }
    
    if dataset_type not in dataset_classes:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Must be one of {list(dataset_classes.keys())}")
    
    # Create dataset with unified interface
    dataset_class = dataset_classes[dataset_type]
    dataset = dataset_class(
        num_images=num_images,
        resolution=resolution,
        radius=radius,
        transform=transform,
        **kwargs
    )
    
    return dataset

def visualize_dataset_samples(dataset, dataset_type, num_samples=8, save_path=None):
    """
    Visualize samples from a dataset to verify correctness.
    
    Parameters:
    - dataset: Dataset object
    - dataset_type: str, type of dataset for title
    - num_samples: int, number of samples to visualize
    - save_path: str, optional path to save the visualization
    """
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img, labels = dataset[i]
        
        # Convert tensor to numpy for visualization
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = np.array(img)
        
        # Normalize to [0, 1] if needed
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"{labels['caption']}", fontsize=10, wrap=True)
        axes[i].axis('off')
        
        # Print label information for debugging
        print(f"Sample {i}: {labels}")
    
    plt.suptitle(f"{dataset_type} Dataset Samples", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def analyze_dataset_statistics(dataset, dataset_type, num_samples=1000):
    """
    Analyze statistical properties of the dataset.
    
    Parameters:
    - dataset: Dataset object
    - dataset_type: str, type of dataset
    - num_samples: int, number of samples to analyze
    """
    
    print(f"\n=== {dataset_type} Dataset Statistics ===")
    
    captions = []
    shapes = []
    locations = []
    
    for i in range(min(num_samples, len(dataset))):
        img, labels = dataset[i]
        
        captions.append(labels['caption'])
        
        # Collect shape and location data
        if 'shape' in labels:  # Single dataset
            shapes.append(labels['shape'])
            locations.append(labels['location'].numpy())
        elif 'shape1' in labels:  # Double/Mixed dataset
            shapes.append(labels['shape1'])
            if labels['shape2'] != -1:  # Not a single object in mixed
                shapes.append(labels['shape2'])
            locations.append(labels['location1'].numpy())
            if labels['location2'][0] != -1:  # Valid location
                locations.append(labels['location2'].numpy())
    
    # Analyze captions
    unique_captions = set(captions)
    print(f"Total samples analyzed: {len(captions)}")
    print(f"Unique captions: {len(unique_captions)}")
    print(f"Caption diversity: {len(unique_captions)/len(captions)*100:.1f}%")
    
    # Show some example captions
    print(f"\nSample captions:")
    for i, caption in enumerate(captions[:5]):
        print(f"  {i+1}: {caption}")
    
    # Analyze shapes
    if shapes:
        shape_counts = {}
        for shape in shapes:
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        
        print(f"\nShape distribution:")
        for shape, count in shape_counts.items():
            print(f"  Shape {shape}: {count} ({count/len(shapes)*100:.1f}%)")
    
    # Analyze locations
    if locations:
        locations = np.array(locations)
        print(f"\nLocation statistics:")
        print(f"  X range: [{locations[:, 0].min():.1f}, {locations[:, 0].max():.1f}]")
        print(f"  Y range: [{locations[:, 1].min():.1f}, {locations[:, 1].max():.1f}]")
        print(f"  Mean X: {locations[:, 0].mean():.1f}")
        print(f"  Mean Y: {locations[:, 1].mean():.1f}")
    
    # For Mixed dataset, analyze type distribution
    if dataset_type == "Mixed":
        types = []
        for i in range(min(num_samples, len(dataset))):
            img, labels = dataset[i]
            if 'type' in labels:
                types.append(labels['type'])
        
        if types:
            type_counts = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            
            print(f"\nType distribution in Mixed dataset:")
            for t, count in type_counts.items():
                type_name = "single" if t == 0 else "double"
                print(f"  {type_name}: {count} ({count/len(types)*100:.1f}%)")

def test_dataset_consistency(dataset, dataset_type):
    """
    Test that the dataset behaves consistently across multiple calls.
    """
    
    print(f"\n=== {dataset_type} Dataset Consistency Test ===")
    
    # Test that same index returns same result (for deterministic datasets)
    img1, labels1 = dataset[0]
    img2, labels2 = dataset[0]
    
    if isinstance(img1, torch.Tensor) and isinstance(img2, torch.Tensor):
        img_consistent = torch.allclose(img1, img2)
    else:
        img_consistent = np.array_equal(img1, img2)
    
    print(f"Same index returns same image: {img_consistent}")
    print(f"Same index returns same labels: {labels1 == labels2}")
    
    # Test that different indices return different results
    img3, labels3 = dataset[1]
    if isinstance(img1, torch.Tensor) and isinstance(img3, torch.Tensor):
        img_different = not torch.allclose(img1, img3)
    else:
        img_different = not np.array_equal(img1, img3)
    
    print(f"Different indices return different images: {img_different}")
    print(f"Different indices return different labels: {labels1 != labels3}")

def comprehensive_test():
    """
    Run comprehensive tests on all dataset types.
    """
    
    print("Starting comprehensive dataset testing...")
    
    # Test parameters
    test_params = {
        "num_images": 100,
        "resolution": 64,
        "radius": 8
    }
    
    # Create output directory for visualizations
    os.makedirs("dataset_testing_output", exist_ok=True)
    
    # Test each dataset type
    for dataset_type in ["Single", "Double", "Mixed"]:
        print(f"\n{'='*60}")
        print(f"Testing {dataset_type} Dataset")
        print(f"{'='*60}")
        
        # Create dataset
        if dataset_type == "Mixed":
            dataset = create_dataset(dataset_type, **test_params, single_ratio=0.3)
        else:
            dataset = create_dataset(dataset_type, **test_params)
        
        # Basic tests
        print(f"Dataset length: {len(dataset)}")
        
        # Test a few samples
        for i in range(3):
            img, labels = dataset[i]
            print(f"Sample {i}: {labels['caption']}")
            if 'type' in labels:
                print(f"  Type: {labels['type']}")
        
        # Visualize samples
        save_path = f"dataset_testing_output/{dataset_type.lower()}_samples.png"
        visualize_dataset_samples(dataset, dataset_type, save_path=save_path)
        
        # Analyze statistics
        analyze_dataset_statistics(dataset, dataset_type)
        
        # Test consistency
        test_dataset_consistency(dataset, dataset_type)
    
    print(f"\n{'='*60}")
    print("Comprehensive testing completed!")
    print("Check the 'dataset_testing_output' directory for visualizations.")

def quick_test():
    """
    Quick test to verify the unified interface works.
    """
    
    print("Quick dataset interface test...")
    
    # Test parameters
    params = {
        "num_images": 50,
        "resolution": 64,
        "radius": 8
    }
    
    # Test all dataset types
    for dataset_type in ["Single", "Double", "Mixed"]:
        print(f"\n--- Testing {dataset_type} Dataset ---")
        
        try:
            if dataset_type == "Mixed":
                dataset = create_dataset(dataset_type, **params, single_ratio=0.3)
            else:
                dataset = create_dataset(dataset_type, **params)
            
            print(f"✓ {dataset_type} dataset created successfully")
            print(f"  Length: {len(dataset)}")
            
            # Test one sample
            img, labels = dataset[0]
            print(f"  Sample caption: {labels['caption']}")
            print(f"  Image shape: {img.shape}")
            print(f"  Label keys: {list(labels.keys())}")
            
        except Exception as e:
            print(f"✗ Error creating {dataset_type} dataset: {e}")
    
    print("\nQuick test completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the unified dataset interface")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--visualize", action="store_true", help="Include visualizations")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        comprehensive_test() 