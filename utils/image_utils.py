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