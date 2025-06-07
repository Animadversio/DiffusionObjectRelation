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


import pandas as pd
import numpy as np

def color_score(detected_rgb, target_rgb):
    max_dist = np.linalg.norm(np.array([255, 255, 255]))
    dist = np.linalg.norm(detected_rgb - target_rgb)
    return max(0, 1 - dist / max_dist)

COLOR_NAME_TO_RGB = {
    'red': np.array([255, 0, 0]),
    'blue': np.array([0, 0, 255]),
    'green': np.array([0, 255, 0]),
    'yellow': np.array([255, 255, 0]),
}

# synonym mapping so "square" and "rectangle" are interchangeable
SHAPE_SYNONYMS = {
    'square': ['square', 'rectangle'],
    'rectangle': ['rectangle', 'square']
}

def evaluate_alignment(prompt, df, color_map=COLOR_NAME_TO_RGB):
    import re
    # parse prompt
    pattern = r'(\w+)\s+(\w+)\s+is\s+to\s+the\s+(left|right|above|below)\s+of\s+(\w+)\s+(\w+)'
    m = re.match(pattern, prompt.lower())
    if not m:
        raise ValueError(f"Prompt '{prompt}' not in expected format")
    color1, obj1, relation, color2, obj2 = m.groups()
    
    # 0) check shape existence with synonyms
    shapes_lower = df['Shape'].str.lower()
    def exists(target_shape):
        return any(shapes_lower.isin(SHAPE_SYNONYMS.get(target_shape, [target_shape])))
    shape_exists = {
        obj1: exists(obj1),
        obj2: exists(obj2)
    }
    shape_match = all(shape_exists.values())
    
    # prepare color arrays
    df_copy = df.copy()
    df_copy['color_array'] = df_copy['Color (RGB)'].apply(lambda x: np.array(x))
    
    # find matching row given synonyms
    def get_row(target_shape):
        possible = SHAPE_SYNONYMS.get(target_shape, [target_shape])
        return df_copy[df_copy['Shape'].str.lower().isin(possible)].iloc[0]
    
    # 1) color1 + obj1 binding
    if shape_exists[obj1]:
        row1 = get_row(obj1)
        score1 = color_score(row1['color_array'], color_map.get(color1, np.array([0,0,0])))
        match1 = score1 > 0.5
    else:
        score1, match1 = 0.0, False
    
    # 2) color2 + obj2 binding
    if shape_exists[obj2]:
        row2 = get_row(obj2)
        score2 = color_score(row2['color_array'], color_map.get(color2, np.array([0,0,0])))
        match2 = score2 > 0.5
    else:
        score2, match2 = 0.0, False
    
    # 3) spatial_color_relation: check whether the red‐object centroid 
    #    is left/above/etc of the blue‐object centroid – regardless of shape
    # first, score every detected object by how close its RGB is to each prompt color
    rel_map = {
            'left':  (0, lambda a, b: a < b),
            'right': (0, lambda a, b: a > b),
            'above': (1, lambda a, b: a < b),
            'below': (1, lambda a, b: a > b),
        }
    df_copy['score_c1'] = df_copy['color_array'].apply(lambda c: color_score(c, color_map[color1]))
    df_copy['score_c2'] = df_copy['color_array'].apply(lambda c: color_score(c, color_map[color2]))
    # pick the best‐matching objects by color
    row_color1 = df_copy.loc[df_copy['score_c1'].idxmax()]
    row_color2 = df_copy.loc[df_copy['score_c2'].idxmax()]
    axis, cond = rel_map[relation]
    spatial_color_relation = bool(cond(
        row_color1['Center (x, y)'][axis],
        row_color2['Center (x, y)'][axis]
    ))

    # 4) spatial_shape_relation: check whether the circle centroid 
    #    is left/above/etc of the square centroid – regardless of color
    if shape_exists[obj1] and shape_exists[obj2]:
        # safe to call get_row now
        row_shape1 = get_row(obj1)
        row_shape2 = get_row(obj2)
        spatial_shape_relation = bool(
            cond(
                row_shape1['Center (x, y)'][axis],
                row_shape2['Center (x, y)'][axis]
            )
        )
    else:
        spatial_shape_relation = False
    
    # overall score
    overall = (int(shape_match) + score1 + score2 + 
               int(spatial_color_relation) + int(spatial_shape_relation)) / 5
    
    return {
        'shape_exists':          shape_exists,
        'shape_match':           shape_match,
        'color_binding_scores':  {obj1: score1, obj2: score2},
        'color_binding_match':   {obj1: match1, obj2: match2},
        'spatial_color_relation':   spatial_color_relation,
        'spatial_shape_relation':   spatial_shape_relation,
        'overall_score':         overall
    }



#result = evaluate_alignment("red circle is to the left of blue square", df)
#print(result)
