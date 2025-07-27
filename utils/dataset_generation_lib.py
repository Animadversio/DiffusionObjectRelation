"""
dataset_generation_lib.py

Class definitions for different types of synthetic datasets and their corresponding 
data loaders.Types of datasets include single-object, double-object, and mixed-object datasets.
All classes utilize stochastic prompts because past experiments show more robust performance. 

Authors: Hannah Kim, Binxu Wang
Date: 2025-07-25
"""

import os
import sys
import json
import random

import numpy as np
import torch
import torch as th
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, TensorDataset

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

from os.path import join

from tqdm.notebook import tqdm, trange

from transformers import T5Tokenizer, T5EncoderModel

from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, ToPILImage, Normalize, CenterCrop

from diffusers import AutoencoderKL

from datasets import load_dataset


class SingleShapeDataset(Dataset):
    def __init__(self, num_images, resolution=64, radius=8, transform=None):
        self.num_images = num_images
        self.shapes = ['triangle', 'circle', 'square']
        self.colors = ['red', 'blue']
        self.articles = ['a', 'an', 'the', 'or', '']
        self.canvas_size = resolution
        self.radius = radius
        self.transform = transform 
        self.shape_to_idx = {'triangle': 0, 'circle': 1, 'square': 2}
        self.color_to_rgb = {'red': 'red', 'blue': 'blue'}

    def __len__(self):
        return self.num_images
    
    def draw_shape_on_image(self, img, shape, location, color):
        draw = ImageDraw.Draw(img)
        x, y = location
        
        if shape == 'circle':
            r = self.radius
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color)
        elif shape == 'square':
            s = self.radius * 2
            draw.rectangle([(x - s//2, y - s//2), (x + s//2, y + s//2)], fill=color)
        elif shape == 'triangle':
            s = self.radius * 2
            h = s * (3 ** 0.5) / 2
            point1 = (x, y - h / 3)
            point2 = (x - s / 2, y + h * 2 / 3)
            point3 = (x + s / 2, y + h * 2 / 3)
            draw.polygon([point1, point2, point3], fill=color)

        return img  

    def __getitem__(self, idx):
        # Blank image
        img = Image.new('RGB', (self.canvas_size, self.canvas_size), 'gray')

        # Randomly select shape and color
        include_color = random.random() < 0.8
        shape = random.choice(self.shapes)
        color = random.choice(self.colors)
        article = random.choice(self.articles)

        components = [article, color, shape]
        caption = f"{article} {color if include_color else ''} {shape.lstrip()}".strip()

        # Random location
        x = random.randint(self.radius + 1, self.canvas_size - self.radius - 1)
        y = random.randint(self.radius + 1, self.canvas_size - self.radius - 1)

        # Draw the shape
        img = self.draw_shape_on_image(img, shape, (x, y), color=self.color_to_rgb[color])

        # Convert image
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        labels = {
            'shape': self.shape_to_idx[shape],
            'location': torch.tensor([x, y], dtype=torch.float32),
            'caption': caption
        }

        return img, labels

class DoubleShapeDataset(Dataset):
    def __init__(self, num_images, resolution=64, radius=16, transform=None):
        """
        Initializes the dataset.

        Parameters:
        - num_images: Integer specifying the number of images in the dataset.
        - transform: Optional torchvision transforms to apply to the images.
        """
        self.num_images = num_images
        self.shapes = ['triangle', 'circle', 'square']
        self.canvas_size = resolution
        self.transform = transform
        self.shape_to_idx = {'triangle': 0, 'circle': 1, 'square': 2}
        self.idx_to_shape = {0: 'triangle', 1: 'circle', 2: 'square'}
        self.radius = radius

        # Spatial relationship phrases for variety
        self.spatial_phrases = {
            'upper_left': ['to the upper left of', 'above and to the left of', 'diagonally up and left from'],
            'upper_right': ['to the upper right of', 'above and to the right of', 'diagonally up and right from'],
            'lower_left': ['to the lower left of', 'below and to the left of', 'diagonally down and left from'],
            'lower_right': ['to the lower right of', 'below and to the right of', 'diagonally down and right from'],
            'above': ['above', 'directly above', 'higher than'],
            'below': ['below', 'directly below', 'lower than'],
            'left': ['to the left of', 'left of', 'left'],
            'right': ['to the right of', 'right of', 'right']
        }

    def generate_caption(self, shape1_idx, shape2_idx, loc1, loc2):
        """
        Generates a natural language caption describing the spatial relationship between two shapes.
        
        Parameters:
        - shape1_idx: Index of first shape
        - shape2_idx: Index of second shape
        - loc1: Coordinates of first shape (x, y)
        - loc2: Coordinates of second shape (x, y)
        
        Returns:
        - string: A natural language caption describing the scene
        """
        # Get shape names
        shape1_name = self.idx_to_shape[shape1_idx]
        shape2_name = self.idx_to_shape[shape2_idx]
        
        # Get coordinates
        x1, y1 = loc1
        x2, y2 = loc2
        
        # Calculate position differences
        dx = x1 - x2  # Positive means shape1 is to the right
        dy = y1 - y2  # Positive means shape1 is lower
        
        # Define thresholds for "directly" above/below/left/right
        threshold = 5  # pixels
        
        # Determine spatial relationship
        if abs(dx) <= threshold:  # Roughly aligned vertically
            if dy < 0:
                relation = random.choice(self.spatial_phrases['above'])
            else:
                relation = random.choice(self.spatial_phrases['below'])
        elif abs(dy) <= threshold:  # Roughly aligned horizontally
            if dx < 0:
                relation = random.choice(self.spatial_phrases['left'])
            else:
                relation = random.choice(self.spatial_phrases['right'])
        else:  # Diagonal relationship
            if dx < 0 and dy < 0:
                relation = random.choice(self.spatial_phrases['upper_left'])
            elif dx < 0 and dy > 0:
                relation = random.choice(self.spatial_phrases['lower_left'])
            elif dx > 0 and dy < 0:
                relation = random.choice(self.spatial_phrases['upper_right'])
            else:  # dx > 0 and dy > 0
                relation = random.choice(self.spatial_phrases['lower_right'])
        
        # Construct caption
        caption = f"{shape1_name} is {relation} {shape2_name}"
        return caption

    def __len__(self):
        return self.num_images

    def draw_shape_on_image(self, img, shape, location, color='black'):
        """
        Draws a specified shape at a given location on the provided image.

        Parameters:
        - img: PIL Image object to draw on.
        - shape: String specifying the shape ('triangle', 'circle', 'square').
        - location: Tuple (x, y) specifying the location of the shape's center.

        Returns:
        - img: PIL Image object with the shape drawn on it.
        """
        draw = ImageDraw.Draw(img)
        x, y = location

        if shape == 'circle':
            r = self.radius  # Radius
            leftUpPoint = (x - r, y - r)
            rightDownPoint = (x + r, y + r)
            draw.ellipse([leftUpPoint, rightDownPoint], fill=color)

        elif shape == 'square':
            s = self.radius * 2  # Side length
            leftUpPoint = (x - s // 2, y - s // 2)
            rightDownPoint = (x + s // 2, y + s // 2)
            draw.rectangle([leftUpPoint, rightDownPoint], fill=color)

        elif shape == 'triangle':
            s = self.radius * 2  # Side length
            h = s * (3 ** 0.5) / 2  # Height of the equilateral triangle
            point1 = (x, y - h / 3)
            point2 = (x - s / 2, y + h * 2 / 3)
            point3 = (x + s / 2, y + h * 2 / 3)
            draw.polygon([point1, point2, point3], fill=color)

        else:
            raise ValueError("Shape must be 'triangle', 'circle', or 'square'.")

        return img

    def __getitem__(self, idx):
        """
        Generates one image and its labels.

        Parameters:
        - idx: Index of the image (not used as images are generated on-the-fly).

        Returns:
        - img: Tensor representing the image.
        - labels: Dictionary containing the shapes and locations of the objects.
        """
        # Create a blank image
        img = Image.new('RGB', (self.canvas_size, self.canvas_size), 'gray')

        # Randomly select two shapes, make sure they are different
        shape1 = random.choice(self.shapes)
        while True:
            shape2 = random.choice(self.shapes)
            if shape1 != shape2:
                break

        # Randomly select locations
        x1 = random.randint(self.radius + 1, self.canvas_size - self.radius - 1)
        y1 = random.randint(self.radius + 1, self.canvas_size - self.radius - 1)
        x2 = random.randint(self.radius + 1, self.canvas_size - self.radius - 1)
        y2 = random.randint(self.radius + 1, self.canvas_size - self.radius - 1)

        # Randomly decide drawing order to allow overlapping
        if random.random() < 0.5:
            img = self.draw_shape_on_image(img, shape1, (x1, y1), color="red")
            img = self.draw_shape_on_image(img, shape2, (x2, y2), color="blue")
            shapes_order = [shape1, shape2]
            locations_order = [(x1, y1), (x2, y2)]
        else:
            img = self.draw_shape_on_image(img, shape2, (x2, y2), color="blue")
            img = self.draw_shape_on_image(img, shape1, (x1, y1), color="red")
            shapes_order = [shape2, shape1]
            locations_order = [(x2, y2), (x1, y1)]

        # Apply transforms if any
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # Encode labels and generate caption
        shape1_idx = self.shape_to_idx[shapes_order[0]]
        shape2_idx = self.shape_to_idx[shapes_order[1]]
        location1 = torch.tensor(locations_order[0], dtype=torch.float32)
        location2 = torch.tensor(locations_order[1], dtype=torch.float32)
        
        caption = self.generate_caption(shape1_idx, shape2_idx, locations_order[0], locations_order[1])

        labels = {
            'shape1': shape1_idx,
            'location1': location1,
            'shape2': shape2_idx,
            'location2': location2,
            'caption': caption
        }

        return img, labels
        

"""

SingleShapeDataset: single_dataset = None, double_dataset = None, single_ratio=1, total_length=10000
DoubleShapeDataset: single_dataset = None, double_dataset = None, single_ratio=0, total_length=10000
MixedShapesDataset: single_dataset, double_dataset, single_ratio, total_length
"""

class MixedShapesDataset(Dataset):
    def __init__(self, single_dataset, double_dataset, single_ratio=0.3, total_length=10000):
        self.single_dataset = single_dataset
        self.double_dataset = double_dataset
        self.single_ratio = single_ratio 

        if total_length is None:
            self.length = len(single_dataset) + len(double_dataset)
        else:
            self.length = total_length 

        num_single = int(self.length * self.single_ratio)
        num_double = self.length - num_single
        self.sample_types = ['single'] * num_single + ['double'] * num_double
        random.shuffle(self.sample_types)

        self.single_indices = list(range(len(single_dataset)))
        self.double_indices = list(range(len(double_dataset)))
        random.shuffle(self.single_indices)
        random.shuffle(self.double_indices)

        self.single_ptr = 0
        self.double_ptr = 0
        self.type_to_idx = {'single': 0, 'double': 1,}
        self.idx_to_type = {0: 'single', 1: 'double'}

    def __len__(self):
        return self.length 

    def __getitem__(self, idx):
        sample_type = self.sample_types[idx]
        if sample_type == 'single':
            if self.single_ptr >= len(self.single_indices):
                self.single_ptr = 0
                random.shuffle(self.single_indices)
            img, labels = self.single_dataset[self.single_indices[self.single_ptr]]
            self.single_ptr += 1

         # Convert to double-object format for consistency
            new_labels = {
                'shape1': labels['shape'],
                'location1': labels['location'],
                'shape2': -1,
                'location2': torch.tensor([-1, -1]),
                'caption': labels['caption'],
                'type': self.type_to_idx['single']
            }
            return img, new_labels
        else:
            if self.double_ptr >= len(self.double_indices):
                self.double_ptr = 0
                random.shuffle(self.double_indices)
            img, labels = self.double_dataset[self.double_indices[self.double_ptr]]
            self.double_ptr += 1

            #Adding type field for consistency 
            labels['type'] = self.type_to_idx['double']
            return img, labels


class ShapesDatasetCached(Dataset):
    filename = 'shapes_dataset_multi_pilot1.pth'
    savedir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/Diffusion_ObjectRelation"

    def __init__(self, transform=None):
        '''
        Initializes the dataset. 

        Parameters:
        - transform: Optional torchvision transforms to apply to the images
        '''

        self.transform = transform
        self.data = torch.load(join(self.savedir, 'dataset', self.filename))
        self.images = self.data['images']
        self.shape1 = self.data['shape1']
        self.location1 = self.data['location1']
        self.shape2 = self.data['shape2']
        self.location2 = self.data['location2']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        shape1 = self.shape1[idx]
        location1 = self.location1[idx]
        shape2 = self.shape2[idx]
        location2 = self.location2[idx]

        labels = {
            'shape1': shape1,
            'location1': location1,
            'shape2': shape2,
            'location2': location2
        }

        if self.transform:
            img = self.transform(img)

        # return img, (shape1, location1, shape2, location2)
        return img, labels 
