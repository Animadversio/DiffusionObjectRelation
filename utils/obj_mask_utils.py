
"""
Object Mask Utilities for Spatial and Semantic Analysis

Tools for generating and manipulating object masks based on spatial relationships 
(e.g., left/right/top/bottom object masks) and shape/color criteria. Useful for 
segmentation and region-based analysis.

Features:
- Spatial Mask Generation:
  * get_left_obj_pos_right_obj_neg_mask(obj_df, object_masks) -> (pos_mask, neg_mask)
  * get_right_obj_pos_left_obj_neg_mask(obj_df, object_masks) -> (pos_mask, neg_mask)
  * get_top_obj_pos_bottom_obj_neg_mask(obj_df, object_masks) -> (pos_mask, neg_mask)
  * get_top_obj_pos_others_neg_mask(obj_df, object_masks) -> (pos_mask, neg_mask)
  * get_bottom_obj_pos_others_neg_mask(obj_df, object_masks) -> (pos_mask, neg_mask)

- Shape-based Masks:
  * get_triangle_pos_others_neg_mask(obj_df, object_masks) -> (pos_mask, neg_mask)
  * get_circle_pos_others_neg_mask(obj_df, object_masks) -> (pos_mask, neg_mask)
  * get_square_pos_others_neg_mask(obj_df, object_masks) -> (pos_mask, neg_mask)

- Color-Shape Combined Masks:
  * get_red_triangle_pos_others_neg_mask(obj_df, object_masks) -> (pos_mask, neg_mask)
  * get_blue_square_pos_others_neg_mask(obj_df, object_masks) -> (pos_mask, neg_mask)
  * And other color-shape combinations...

Author: Binxu
"""

import os
from os.path import join
import pickle
import cv2
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from PIL import Image
from matplotlib import pyplot as plt
from tqdm.auto import trange, tqdm
import sys
sys.path.append("/n/home13/xupan/sompolinsky_lab/DiffusionObjectRelation")
sys.path.append("/n/home12/binxuwang/Github/DiffusionObjectRelation")
from utils.cv2_eval_utils import find_classify_object_masks

positive_threshold = 180 
MAP_SHAPE = (8, 8)

def get_left_obj_pos_right_obj_neg_mask(obj_df, object_masks):
    """
    Left object is positive, right object is negative.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    
    if obj_df.iloc[0]['Center (x, y)'][0] < obj_df.iloc[1]['Center (x, y)'][0]:
        positive_mask = object_masks[0]
        negative_mask = object_masks[1]
    else:
        positive_mask = object_masks[1]
        negative_mask = object_masks[0]
    return positive_mask, negative_mask


def get_right_obj_pos_left_obj_neg_mask(obj_df, object_masks):
    """
    Right object is positive, left object is negative.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    
    if obj_df.iloc[0]['Center (x, y)'][0] > obj_df.iloc[1]['Center (x, y)'][0]:
        positive_mask = object_masks[0]
        negative_mask = object_masks[1]
    else:
        positive_mask = object_masks[1]
        negative_mask = object_masks[0]
    return positive_mask, negative_mask


def get_left_obj_pos_others_neg_mask(obj_df, object_masks):
    """
    Left object is positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    if obj_df.iloc[0]['Center (x, y)'][0] < obj_df.iloc[1]['Center (x, y)'][0]:
        positive_mask = object_masks[0]
        negative_mask = ~object_masks[0]
    else:
        positive_mask = object_masks[1]
        negative_mask = ~object_masks[1]
    return positive_mask, negative_mask


def get_right_obj_pos_others_neg_mask(obj_df, object_masks):
    """
    Right object is positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    if obj_df.iloc[0]['Center (x, y)'][0] > obj_df.iloc[1]['Center (x, y)'][0]:
        positive_mask = object_masks[0]
        negative_mask = ~object_masks[0]
    else:
        positive_mask = object_masks[1]
        negative_mask = ~object_masks[1]
    return positive_mask, negative_mask


def get_top_obj_pos_bottom_obj_neg_mask(obj_df, object_masks):
    """
    Top object is positive, bottom object is negative.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    
    if obj_df.iloc[0]['Center (x, y)'][1] < obj_df.iloc[1]['Center (x, y)'][1]:
        positive_mask = object_masks[0]
        negative_mask = object_masks[1]
    else:
        positive_mask = object_masks[1]
        negative_mask = object_masks[0]
    return positive_mask, negative_mask


def get_top_obj_pos_others_neg_mask(obj_df, object_masks):
    """
    Top object is positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    mask_dtype = object_masks[0].dtype
    if obj_df.iloc[0]['Center (x, y)'][1] < obj_df.iloc[1]['Center (x, y)'][1]:
        positive_mask = object_masks[0]
    else:
        positive_mask = object_masks[1]
    if mask_dtype == bool:
        negative_mask = ~positive_mask
    else:
        negative_mask = np.clip(1 - positive_mask, 0, 1)
    return positive_mask, negative_mask


def get_bottom_obj_pos_others_neg_mask(obj_df, object_masks):
    """
    Bottom object is positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    mask_dtype = object_masks[0].dtype
    if obj_df.iloc[0]['Center (x, y)'][1] > obj_df.iloc[1]['Center (x, y)'][1]:
        positive_mask = object_masks[0]
    else:
        positive_mask = object_masks[1]
    if mask_dtype == bool:
        negative_mask = ~positive_mask
    else:
        negative_mask = np.clip(1 - positive_mask, 0, 1)
    return positive_mask, negative_mask


def get_triangle_pos_others_neg_mask(obj_df, object_masks):
    """
    Triangle objects are positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    mask_dtype = object_masks[0].dtype
    positive_token_mask = np.zeros_like(object_masks[0], dtype=mask_dtype)
    for i in range(len(obj_df)):
        if obj_df.iloc[i]['Shape'] == "Triangle":
            if mask_dtype == bool:
                positive_token_mask = positive_token_mask | object_masks[i]
            else:
                positive_token_mask = positive_token_mask + object_masks[i]

    positive_mask = positive_token_mask
    if mask_dtype == bool:
        negative_mask = ~positive_token_mask
    else:
        negative_mask = np.clip(1 - positive_token_mask, 0, 1)
    return positive_mask, negative_mask


def get_circle_pos_others_neg_mask(obj_df, object_masks):
    """
    Circle objects are positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    mask_dtype = object_masks[0].dtype
    positive_token_mask = np.zeros_like(object_masks[0], dtype=mask_dtype)
    for i in range(len(obj_df)):
        if obj_df.iloc[i]['Shape'] == "Circle":
            if mask_dtype == bool:
                positive_token_mask = positive_token_mask | object_masks[i]
            else:
                positive_token_mask = positive_token_mask + object_masks[i]

    positive_mask = positive_token_mask
    if mask_dtype == bool:
        negative_mask = ~positive_token_mask
    else:
        negative_mask = np.clip(1 - positive_token_mask, 0, 1)
    return positive_mask, negative_mask


def get_square_pos_others_neg_mask(obj_df, object_masks):
    """
    Square objects are positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    
    mask_dtype = object_masks[0].dtype
    positive_token_mask = np.zeros_like(object_masks[0], dtype=mask_dtype)
    for i in range(len(obj_df)):
        if obj_df.iloc[i]['Shape'] == "Square":
            if mask_dtype == bool:
                positive_token_mask = positive_token_mask | object_masks[i]
            else:
                positive_token_mask = positive_token_mask + object_masks[i]

    positive_mask = positive_token_mask
    if mask_dtype == bool:
        negative_mask = ~positive_token_mask
    else:
        negative_mask = np.clip(1 - positive_token_mask, 0, 1)
    return positive_mask, negative_mask


def get_red_triangle_pos_others_neg_mask(obj_df, object_masks):
    """
    Red triangle objects are positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    
    mask_dtype = object_masks[0].dtype
    positive_token_mask = np.zeros_like(object_masks[0], dtype=mask_dtype)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if obj_df.iloc[i]['Shape'] == "Triangle" and Rvalue > 225 and Gvalue < 30 and Bvalue < 30:
            if mask_dtype == bool:
                positive_token_mask = positive_token_mask | object_masks[i]
            else:
                positive_token_mask = positive_token_mask + object_masks[i]
            
    positive_mask = positive_token_mask
    if mask_dtype == bool:
        negative_mask = ~positive_token_mask
    else:
        negative_mask = np.clip(1 - positive_token_mask, 0, 1)
    return positive_mask, negative_mask


def get_red_square_pos_others_neg_mask(obj_df, object_masks):
    """
    Red squares are positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    
    mask_dtype = object_masks[0].dtype
    positive_token_mask = np.zeros_like(object_masks[0], dtype=mask_dtype)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if obj_df.iloc[i]['Shape'] == "Square" and Rvalue > 225 and Gvalue < 30 and Bvalue < 30:
            if mask_dtype == bool:
                positive_token_mask = positive_token_mask | object_masks[i]
            else:
                positive_token_mask = positive_token_mask + object_masks[i]
            
    positive_mask = positive_token_mask
    if mask_dtype == bool:
        negative_mask = ~positive_token_mask
    else:
        negative_mask = np.clip(1 - positive_token_mask, 0, 1)
    return positive_mask, negative_mask


def get_red_circle_pos_others_neg_mask(obj_df, object_masks):
    """
    Red circles are positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    
    mask_dtype = object_masks[0].dtype
    positive_token_mask = np.zeros_like(object_masks[0], dtype=mask_dtype)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if obj_df.iloc[i]['Shape'] == "Circle" and Rvalue > 225 and Gvalue < 30 and Bvalue < 30:
            if mask_dtype == bool:
                positive_token_mask = positive_token_mask | object_masks[i]
            else:
                positive_token_mask = positive_token_mask + object_masks[i]
            
    positive_mask = positive_token_mask
    if mask_dtype == bool:
        negative_mask = ~positive_token_mask
    else:
        negative_mask = np.clip(1 - positive_token_mask, 0, 1)
    return positive_mask, negative_mask


def get_blue_triangle_pos_others_neg_mask(obj_df, object_masks):
    """
    Blue triangles are positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    
    mask_dtype = object_masks[0].dtype
    positive_token_mask = np.zeros_like(object_masks[0], dtype=mask_dtype)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if obj_df.iloc[i]['Shape'] == "Triangle" and Rvalue < 30 and Gvalue < 30 and Bvalue > 225:
            if mask_dtype == bool:
                positive_token_mask = positive_token_mask | object_masks[i]
            else:
                positive_token_mask = positive_token_mask + object_masks[i]
            
    positive_mask = positive_token_mask
    if mask_dtype == bool:
        negative_mask = ~positive_token_mask
    else:
        negative_mask = np.clip(1 - positive_token_mask, 0, 1)
    return positive_mask, negative_mask


def get_blue_square_pos_others_neg_mask(obj_df, object_masks):
    """
    Blue squares are positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    
    mask_dtype = object_masks[0].dtype
    positive_token_mask = np.zeros_like(object_masks[0], dtype=mask_dtype)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if obj_df.iloc[i]['Shape'] == "Square" and Rvalue < 30 and Gvalue < 30 and Bvalue > 225:
            if mask_dtype == bool:
                positive_token_mask = positive_token_mask | object_masks[i]
            else:
                positive_token_mask = positive_token_mask + object_masks[i]
            
    positive_mask = positive_token_mask
    if mask_dtype == bool:
        negative_mask = ~positive_token_mask
    else:
        negative_mask = np.clip(1 - positive_token_mask, 0, 1)
    return positive_mask, negative_mask


def get_blue_circle_pos_others_neg_mask(obj_df, object_masks):
    """
    Blue circles are positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    
    mask_dtype = object_masks[0].dtype
    positive_token_mask = np.zeros_like(object_masks[0], dtype=mask_dtype)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if obj_df.iloc[i]['Shape'] == "Circle" and Rvalue < 30 and Gvalue < 30 and Bvalue > 225:
            if mask_dtype == bool:
                positive_token_mask = positive_token_mask | object_masks[i]
            else:
                positive_token_mask = positive_token_mask + object_masks[i]
            
    positive_mask = positive_token_mask
    if mask_dtype == bool:
        negative_mask = ~positive_token_mask
    else:
        negative_mask = np.clip(1 - positive_token_mask, 0, 1)
    return positive_mask, negative_mask


def get_red_obj_pos_others_neg_mask(obj_df, object_masks):
    """
    Red objects are positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    
    positive_token_mask = np.zeros_like(object_masks[0], dtype=bool)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if Rvalue > 225 and Gvalue < 30 and Bvalue < 30:
            positive_token_mask = positive_token_mask | object_masks[i]
            
    positive_mask = positive_token_mask
    negative_mask = ~positive_token_mask
    return positive_mask, negative_mask


def get_blue_obj_pos_others_neg_mask(obj_df, object_masks):
    """
    Blue objects are positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    
    positive_token_mask = np.zeros_like(object_masks[0], dtype=bool)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if Rvalue < 30 and Gvalue < 30 and Bvalue > 225:
            positive_token_mask = positive_token_mask | object_masks[i]
            
    positive_mask = positive_token_mask
    negative_mask = ~positive_token_mask
    return positive_mask, negative_mask


def get_obj_pos_others_neg_mask(obj_df, object_masks):
    """
    All objects are positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    
    positive_token_mask = np.zeros_like(object_masks[0], dtype=bool)
    for i in range(len(obj_df)):
        positive_token_mask = positive_token_mask | object_masks[i]
            
    positive_mask = positive_token_mask
    negative_mask = ~positive_token_mask
    return positive_mask, negative_mask


def get_background_pos_obj_neg_mask(obj_df, object_masks):
    """
    All objects are positive, others are negative including background.
    Returns positive mask and negative mask.
    """
    
    if len(obj_df) == 0 or object_masks is None or len(object_masks) == 0:
        return np.zeros(MAP_SHAPE, dtype=float), np.zeros(MAP_SHAPE, dtype=float)
    elif len(obj_df) != 2 or len(object_masks) != 2:
        # Handle when not exactly 2 objects
        mask_dtype = object_masks[0].dtype if len(object_masks) > 0 else float
        return np.zeros_like(object_masks[0], dtype=mask_dtype), np.zeros_like(object_masks[0], dtype=mask_dtype)
    
    mask_dtype = object_masks[0].dtype
    object_token_mask = np.zeros_like(object_masks[0], dtype=mask_dtype)
    for i in range(len(obj_df)):
        if mask_dtype == bool:
            object_token_mask = object_token_mask | object_masks[i]
        else:
            object_token_mask = object_token_mask + object_masks[i]
            
    negative_mask = object_token_mask
    if mask_dtype == bool:
        positive_mask = ~object_token_mask
    else:
        positive_mask = np.clip(1 - object_token_mask, 0, 1)
    return positive_mask, negative_mask
