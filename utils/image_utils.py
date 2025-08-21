"""
Image Processing Utilities

Tools for arranging PIL images into grids, resizing, normalizing, and converting
between tensor and image formats for visualization.

Features:
- Grid Creation:
  * pil_images_to_grid(images, grid_size=None, image_size=None, padding=2, 
                       normalize=False, background_color=(255,255,255)) -> PIL.Image

- Panel Extraction:
  * extract_panel_from_grid(grid_image, row, col, grid_size, padding=2) -> PIL.Image
  * get_grid_info(grid_image, grid_size, padding=2) -> dict

Author: Binxu
"""

import math
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils

def pil_images_to_grid(
    images,
    grid_size=None,
    image_size=None,
    padding=2,
    normalize=False,
    background_color=(255, 255, 255)
):
    """
    Arrange a list of PIL Images into a grid and return the grid as a PIL Image.

    :param images: List of PIL.Image objects.
    :param grid_size: Tuple (columns, rows). If None, grid size is calculated to be as square as possible.
    :param image_size: Tuple (width, height). If provided, images will be resized to this size.
    :param padding: Padding between images in pixels.
    :param normalize: Whether to normalize the images.
    :param background_color: Background color tuple, e.g., (255, 255, 255) for white.
    :return: A PIL.Image object representing the grid.
    """
    if not images:
        raise ValueError("The images list is empty.")

    num_images = len(images)

    # Determine grid size
    if grid_size:
        cols, rows = grid_size
    else:
        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)

    # Define transformation: resize and convert to tensor
    transform_list = []
    if image_size:
        transform_list.append(transforms.Resize(image_size))
    transform_list.append(transforms.ToTensor())  # Converts to [0,1] range
    transform = transforms.Compose(transform_list)

    # Apply transformations
    transformed_images = []
    for img in images:
        img = img.convert('RGB')  # Ensure all images have 3 channels
        img = transform(img)
        transformed_images.append(img)

    # Stack images into a single tensor
    batch_tensor = torch.stack(transformed_images)  # Shape: (B, C, H, W)

    # Create grid using torchvision
    grid_tensor = vutils.make_grid(
        batch_tensor,
        nrow=cols,
        padding=padding,
        normalize=normalize,
        pad_value=0  # Will set background color later
    )

    # Convert grid tensor to PIL Image
    # If normalize is True, make_grid scales the tensor to [0,1], else [0,1] assuming input was [0,1]
    to_pil = transforms.ToPILImage()
    grid_image = to_pil(grid_tensor)

    # If padding is set and background_color is not white, adjust the padding
    if padding > 0 and background_color != (0, 0, 0):
        # Create a new image with the desired background color
        grid_width, grid_height = grid_image.size
        bg = Image.new('RGB', grid_image.size, background_color)
        bg.paste(grid_image, mask=grid_image.split()[3] if grid_image.mode == 'RGBA' else None)
        grid_image = bg

    return grid_image


def extract_panel_from_grid(grid_image, row, col, grid_size, padding=2):
    """
    Extract a specific panel from a PIL image grid.
    
    Args:
        grid_image: PIL.Image - The grid image to extract from
        row: int - Row index of the panel to extract (0-indexed)
        col: int - Column index of the panel to extract (0-indexed)
        grid_size: tuple - (cols, rows) - Number of columns and rows in the grid
        padding: int - Padding between panels in pixels (default: 2)
        
    Returns:
        PIL.Image - The extracted panel
        
    Example:
        >>> grid = pil_images_to_grid(images, grid_size=(5, 5))
        >>> panel = extract_panel_from_grid(grid, row=0, col=0, grid_size=(5, 5))
    """
    if not isinstance(grid_image, Image.Image):
        raise ValueError("grid_image must be a PIL Image")
    
    cols, rows = grid_size
    
    if row < 0 or row >= rows:
        raise ValueError(f"Row index {row} out of range for grid with {rows} rows")
    if col < 0 or col >= cols:
        raise ValueError(f"Column index {col} out of range for grid with {cols} columns")
    
    # Get grid dimensions
    grid_width, grid_height = grid_image.size
    
    # Calculate panel dimensions (accounting for padding)
    # The grid has padding between panels and around the edges
    total_padding_width = padding * (cols + 1)  # padding on left/right of each column plus edges
    total_padding_height = padding * (rows + 1)  # padding on top/bottom of each row plus edges
    
    panel_width = (grid_width - total_padding_width) // cols
    panel_height = (grid_height - total_padding_height) // rows
    
    # Calculate the position of the panel in the grid
    # Starting position includes edge padding plus previous panels and their padding
    left = padding + col * (panel_width + padding)
    top = padding + row * (panel_height + padding)
    right = left + panel_width
    bottom = top + panel_height
    
    # Extract the panel using PIL crop
    panel = grid_image.crop((left, top, right, bottom))
    
    return panel


def get_grid_info(grid_image, grid_size, padding=2):
    """
    Get information about a grid image including panel dimensions and positions.
    
    Args:
        grid_image: PIL.Image - The grid image
        grid_size: tuple - (cols, rows) - Number of columns and rows in the grid
        padding: int - Padding between panels in pixels (default: 2)
        
    Returns:
        dict - Information about the grid:
            - panel_width: Width of each panel
            - panel_height: Height of each panel
            - total_panels: Total number of panels
            - positions: List of (left, top, right, bottom) for each panel
    """
    cols, rows = grid_size
    grid_width, grid_height = grid_image.size
    
    # Calculate panel dimensions
    total_padding_width = padding * (cols + 1)
    total_padding_height = padding * (rows + 1)
    
    panel_width = (grid_width - total_padding_width) // cols
    panel_height = (grid_height - total_padding_height) // rows
    
    # Calculate positions for all panels
    positions = []
    for row in range(rows):
        for col in range(cols):
            left = padding + col * (panel_width + padding)
            top = padding + row * (panel_height + padding)
            right = left + panel_width
            bottom = top + panel_height
            positions.append((left, top, right, bottom))
    
    return {
        'panel_width': panel_width,
        'panel_height': panel_height,
        'total_panels': cols * rows,
        'positions': positions
    }