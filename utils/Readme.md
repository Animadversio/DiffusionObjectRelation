# Utils Folder Overview

This folder contains a collection of utility modules and libraries used throughout the project. Below is a brief description of each file:

* **Dataset generation and evaluation**
    * **relation_shape_dataset_lib.py**  
    Provides a PyTorch Dataset for generating synthetic images of geometric shapes (triangle, circle, square) with spatial relationships and captions. Useful for training and evaluating models on spatial reasoning tasks.

    * **cv2_eval_utils.py**  
    Contains functions for object detection, classification, and evaluation using OpenCV. Includes utilities for extracting object masks, classifying shapes/colors, and evaluating spatial relationships in images.

    * **obj_mask_utils.py**  
    Provides utilities for generating and manipulating object masks based on spatial relationships (e.g., left/right/top/bottom object masks) and shape/color criteria. Useful for segmentation and region-based analysis.

* **Tools related to Attention Map Recording and Analysis**
    * **attention_map_store_utils.py**  
    Provides classes and functions to store and visualize attention maps during model inference, especially for PixArt models. Includes custom attention processors and hook management.

    * **attention_analysis_lib.py**  
    Offers tools for analyzing and visualizing attention maps in transformers after obtaining them with **attention_map_store_utils.py**, including entropy, spatial variance, and various plotting utilities for attention heads and layers.

    * **mask_attention_utils.py**  
    Utilities for manipulating attention masks in text prompts, including masking specific semantic parts (objects, colors, spatial relations) using spaCy and tokenizers.

* **Tools related to Customization of PixArt Model and Sampling**
    * **pixart_utils.py**  
    Utilities for working with PixArt and DiT (Diffusion Transformer) models, including state dict conversion, model construction, and pipeline setup for image generation.

    * **pixart_sampling_utils.py**  
    Contains functions for sampling and visualizing images from PixArt pipelines, including prompt embedding, latent trajectory visualization, and custom pipeline classes for advanced inference.

    * **text_encoder_control_lib.py**  
    Implements custom text encoder classes (random embedding, positional encoding) and utilities for saving and managing prompt embeddings, primarily for use with T5 and similar models.

    * **custom_text_encoding_utils.py**  
    Contains custom text encoding classes and functions for tokenizing captions, generating random embeddings, and saving prompt embeddings with or without positional encodings.

* **Tools for recording activations and finding features directions**
    * **find_features_classifier_lib.py**  
    Contains functions for extracting features from images, generating positive/negative embeddings based on object masks, and training classifiers to distinguish between spatial or semantic categories.

    * **layer_hook_utils.py**  
    Provides advanced utilities for registering hooks in PyTorch models to extract activations, layer names, and module information. Useful for model introspection and feature extraction.
* **Basic utils**
    * **plot_utils.py**  
    A collection of plotting utilities for images and grids, including functions to display, save, and arrange images in grids, as well as helper functions for matplotlib-based visualization.

    * **image_utils.py**  
    Utility functions for arranging PIL images into grids, resizing, normalizing, and converting between tensor and image formats. Useful for visualizing batches of images.