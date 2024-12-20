import cv2
import numpy as np
import pandas as pd
from PIL import Image

def find_classify_objects(image, area_threshold=100, radius=16.0):
    if isinstance(image, Image.Image):
        image = np.array(image)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    classified_objects = []
    # go through each color channel
    for channel in range(3):
        gray_image = image[:,:,channel]
        # Threshold the image to create a binary mask
        _, binary_mask = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)
        # Find contours of the shapes
        contours, _ = cv2.findContours(binary_mask, 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
        # Initialize results
        for i, contour in enumerate(contours):
            # Calculate properties of the contour
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            # Shape classification based on the number of vertices
            if len(approx) == 3:
                shape = "Triangle"
                s = radius * 2  # Side length
                h = s * (3 ** 0.5) / 2  # Height of the equilateral triangle
                expected_area = s * h / 2
            elif len(approx) == 4:
                shape = "Square" if abs(w - h) < 5 else "Rectangle"
                s = radius * 2
                expected_area = s**2
            elif len(approx) > 4:
                shape = "Circle"
                expected_area = np.pi * radius ** 2
            else:
                shape = "Unknown"
                expected_area = np.nan
            # Calculate the color of the shape by extracting the region
            mask = np.zeros_like(gray_image)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color = cv2.mean(image, mask=mask)
            # Add to results
            if area < area_threshold:
                continue
            classified_objects.append({
                "Object": i + 1,
                "Shape": shape,
                "Color (RGB)": tuple(map(int, mean_color[:3])),
                "Center (x, y)": (x + w // 2, y + h // 2),
                "Area": area,
                "Expected Area": expected_area
            })

    # Convert to DataFrame for better visualization
    classified_objects_df = pd.DataFrame(classified_objects)
    classified_objects_df
    return classified_objects_df


def find_classify_object_masks(image, area_threshold=100, radius=16.0):
    if isinstance(image, Image.Image):
        image = np.array(image)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    classified_objects = []
    object_masks = []
    # go through each color channel
    for channel in range(3):
        gray_image = image[:,:,channel]
        # Threshold the image to create a binary mask
        _, binary_mask = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)
        # Find contours of the shapes
        contours, _ = cv2.findContours(binary_mask, 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
        # Initialize results
        for i, contour in enumerate(contours):
            # Calculate properties of the contour
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            # Shape classification based on the number of vertices
            if len(approx) == 3:
                shape = "Triangle"
                s = radius * 2  # Side length
                h = s * (3 ** 0.5) / 2  # Height of the equilateral triangle
                expected_area = s * h / 2
            elif len(approx) == 4:
                shape = "Square" if abs(w - h) < 5 else "Rectangle"
                s = radius * 2
                expected_area = s**2
            elif len(approx) > 4:
                shape = "Circle"
                expected_area = np.pi * radius ** 2
            else:
                shape = "Unknown"
                expected_area = np.nan
            # Calculate the color of the shape by extracting the region
            mask = np.zeros_like(gray_image)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color = cv2.mean(image, mask=mask)
            # Add to results
            if area < area_threshold:
                continue
            classified_objects.append({
                "Object": i + 1,
                "Shape": shape,
                "Color (RGB)": tuple(map(int, mean_color[:3])),
                "Center (x, y)": (x + w // 2, y + h // 2),
                "Area": area,
                "Expected Area": expected_area
            })
            object_masks.append(mask)

    # Convert to DataFrame for better visualization
    classified_objects_df = pd.DataFrame(classified_objects)
    assert len(classified_objects_df) == len(object_masks)
    return classified_objects_df, object_masks


import pandas as pd

def identity_spatial_relation(x1, y1, x2, y2):
    dx = x1 - x2  # Positive means shape1 is to the right
    dy = y1 - y2  # Positive means shape1 is lower
    # Define thresholds for "directly" above/below/left/right
    threshold = 5  # pixels
    if abs(dx) <= threshold:  # Roughly aligned vertically
        if dy < 0:
            observed_relation = 'above'
        else:
            observed_relation = 'below'
    elif abs(dy) <= threshold:  # Roughly aligned horizontally
        if dx < 0:
            observed_relation = 'left'
        else:
            observed_relation = 'right'
    else:  # Diagonal relationship
        if dx < 0 and dy < 0:
            observed_relation = 'upper_left'
        elif dx < 0 and dy > 0:
            observed_relation = 'lower_left'
        elif dx > 0 and dy < 0:
            observed_relation = 'upper_right'
        else:  # dx > 0 and dy > 0
            observed_relation = 'lower_right'
    return observed_relation


def evaluate_parametric_relation(df, scene_info, MARGIN=25):
    """ blue_triangle_is_above_red_triangle
    Evaluates if a blue-dominant triangle is above a red-dominant triangle in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing object detection details. It must include 
                       columns 'Shape', 'Color (RGB)', 'Center (x, y)', and 'Area'.

    Returns:
    bool: True if a blue-dominant triangle is above a red-dominant triangle, False otherwise.
    """
    # Validate input
    if df.empty:
        return False, "no object"
    if not all(col in df.columns for col in ['Shape', 'Color (RGB)', 'Center (x, y)']):
        # return False, "no object"
        raise ValueError("DataFrame must contain 'Shape', 'Color (RGB)', and 'Center (x, y)' columns.")
    shape1 = scene_info["shape1"] 
    shape2 = scene_info["shape2"]
    color1 = scene_info["color1"]
    color2 = scene_info["color2"]
    spatial_relationship = scene_info["spatial_relationship"]
    # Extract triangles
    df["is_red"] = df['Color (RGB)'].apply(lambda rgb: rgb[0] > 255-MARGIN and rgb[1] < MARGIN and rgb[2] < MARGIN)
    df["is_blue"] = df['Color (RGB)'].apply(lambda rgb: rgb[2] > 255-MARGIN and rgb[0] < MARGIN and rgb[1] < MARGIN)
    # Identify red-dominant and blue-dominant triangles
    mask1 = np.ones(len(df), dtype=bool)
    if shape1 is not None:
        mask1 = mask1 & (df['Shape'] == shape1)
    if color1 is not None:
        if color1 == "red":
            mask1 = mask1 & (df['is_red'] == True)
        elif color1 == "blue":
            mask1 = mask1 & (df['is_blue'] == True)
    obj1_df = df[mask1]
    if obj1_df.empty:
        return False, "missing object 1"
    
    mask2 = np.ones(len(df), dtype=bool)
    if shape2 is not None:
        mask2 = mask2 & (df['Shape'] == shape2)
    if color2 is not None:
        if color2 == "red":
            mask2 = mask2 & (df['is_red'] == True)
        elif color2 == "blue":
            mask2 = mask2 & (df['is_blue'] == True)
    obj2_df = df[mask2]
    if obj2_df.empty:
        return False, "missing object 2"

    # Compare the y-coordinates (assuming y increases downwards)
    if len(obj1_df) == 1 and len(obj2_df) == 1:
        x1, y1 = obj1_df['Center (x, y)'].iloc[0]
        x2, y2 = obj2_df['Center (x, y)'].iloc[0]
        observed_relation = identity_spatial_relation(x1, y1, x2, y2)
        rel_correct = spatial_relationship == observed_relation
    elif len(obj1_df) == len(obj2_df) == 2 and obj1_df.equals(obj2_df):
        # two objects are the same and the two objects can be in any order
        x1, y1 = obj1_df['Center (x, y)'].iloc[0]
        x2, y2 = obj1_df['Center (x, y)'].iloc[1]
        observed_relation1 = identity_spatial_relation(x1, y1, x2, y2)
        observed_relation2 = identity_spatial_relation(x2, y2, x1, y1)
        rel_correct = spatial_relationship in [observed_relation1, observed_relation2]
    else:
        return False, "number of objects incorrect"
    if rel_correct:
        return True, "correct"
    else:
        return False, "spatial relation incorrect" # and abs(blue_x - red_x) < 10


def eval_func_factory(prompt_name):
    return lambda df: evaluate_parametric_relation(df, scene_info_collection[prompt_name])


scene_info_collection = {'blue_triangle_is_above_red_triangle':  {"color1": "blue", "shape1": "Triangle", "color2": "red", "shape2": "Triangle", "spatial_relationship": "above"},
                        'blue_circle_is_above_and_to_the_right_of_blue_square':  {"color1": "blue", "shape1": "Circle", "color2": "blue", "shape2": "Square", "spatial_relationship": "upper_right"},
                        'blue_circle_is_above_blue_square':  {"color1": "blue", "shape1": "Circle", "color2": "blue", "shape2": "Square", "spatial_relationship": "above"},
                        'blue_square_is_to_the_right_of_red_circle':  {"color1": "blue", "shape1": "Square", "color2": "red", "shape2": "Circle", "spatial_relationship": "right"},
                        'blue_triangle_is_above_red_triangle':  {"color1": "blue", "shape1": "Triangle", "color2": "red", "shape2": "Triangle", "spatial_relationship": "above"},
                        'blue_triangle_is_to_the_upper_left_of_red_square':  {"color1": "blue", "shape1": "Triangle", "color2": "red", "shape2": "Square", "spatial_relationship": "upper_left"},
                        'circle_is_below_red_square':  {"color1": None, "shape1": "Circle", "color2": "red", "shape2": "Square", "spatial_relationship": "below"},
                        'red_circle_is_above_square':  {"color1": "red", "shape1": "Circle", "color2": None, "shape2": "Square", "spatial_relationship": "above"},
                        'red_circle_is_to_the_left_of_blue_square':  {"color1": "red", "shape1": "Circle", "color2": "blue", "shape2": "Square", "spatial_relationship": "left"},
                        'red_is_above_blue':  {"color1": "red", "shape1": None, "color2": "blue", "shape2": None, "spatial_relationship": "above"},  # TODO: check 
                        'red_is_to_the_left_of_red':  {"color1": "red", "shape1": None, "color2": "red", "shape2": None, "spatial_relationship": "left"},  # TODO: check 
                        'triangle_is_above_and_to_the_right_of_square':  {"color1": None, "shape1": "Triangle", "color2": None, "shape2": "Square", "spatial_relationship": "upper_right"},
                        'triangle_is_above_red_circle':  {"color1": None, "shape1": "Triangle", "color2": "red", "shape2": "Circle", "spatial_relationship": "above"},
                        'triangle_is_to_the_left_of_square':  {"color1": None, "shape1": "Triangle", "color2": None, "shape2": "Square", "spatial_relationship": "left"},
                        'triangle_is_to_the_left_of_triangle':  {"color1": None, "shape1": "Triangle", "color2": None, "shape2": "Triangle", "spatial_relationship": "left"},  # TODO: check 
                        'triangle_is_to_the_upper_left_of_square':  {"color1": None, "shape1": "Triangle", "color2": None, "shape2": "Square", "spatial_relationship": "upper_left"},
                        }

