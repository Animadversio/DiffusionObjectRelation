# %% [markdown]
# Use the embedding Binxu generated to find features. Try to find features by classifing contrastive labeled data.
# %% [markdown]
# ## Read embeddings, extract token embedding based on contrastive classes
# %%
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
sys.path.append("/n/home13/xupan/sompolinsky_lab/DiffusionObjectRelation/utils")
from plot_utils import saveallforms
from cv2_eval_utils import find_classify_object_masks

# %% [markdown]
# #### Utils function to modularize positive / negative embedding extraction code

# %%
def get_left_obj_pos_right_obj_neg(latent_state, obj_df, objeect_masks):
    """
    Left object is positive, right object is negative.
    No label for background.
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    if obj_df.iloc[0]['Center (x, y)'][0] < obj_df.iloc[1]['Center (x, y)'][0]:
        positive_embeddings = latent_state[objeect_masks[0], :].numpy()
        negative_embeddings = latent_state[objeect_masks[1], :].numpy()
    else:
        positive_embeddings = latent_state[objeect_masks[1], :].numpy()
        negative_embeddings = latent_state[objeect_masks[0], :].numpy()
    return [positive_embeddings], [negative_embeddings]

def get_right_obj_pos_left_obj_neg(latent_state, obj_df, objeect_masks):
    """
    Right object is positive, left object is negative.
    No label for background.
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    if obj_df.iloc[0]['Center (x, y)'][0] > obj_df.iloc[1]['Center (x, y)'][0]:
        positive_embeddings = latent_state[objeect_masks[0], :].numpy()
        negative_embeddings = latent_state[objeect_masks[1], :].numpy()
    else:
        positive_embeddings = latent_state[objeect_masks[1], :].numpy()
        negative_embeddings = latent_state[objeect_masks[0], :].numpy()
    return [positive_embeddings], [negative_embeddings]

def get_left_obj_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Left object is positive, others are negative including background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    if obj_df.iloc[0]['Center (x, y)'][0] < obj_df.iloc[1]['Center (x, y)'][0]:
        positive_embeddings = latent_state[objeect_masks[0], :].numpy()
        negative_embeddings = latent_state[~objeect_masks[0], :].numpy()
    else:
        positive_embeddings = latent_state[objeect_masks[1], :].numpy()
        negative_embeddings = latent_state[~objeect_masks[1], :].numpy()
    return [positive_embeddings], [negative_embeddings]

def get_right_obj_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Right object is positive, others are negative including background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    if obj_df.iloc[0]['Center (x, y)'][0] > obj_df.iloc[1]['Center (x, y)'][0]:
        positive_embeddings = latent_state[objeect_masks[0], :].numpy()
        negative_embeddings = latent_state[~objeect_masks[0], :].numpy()
    else:
        positive_embeddings = latent_state[objeect_masks[1], :].numpy()
        negative_embeddings = latent_state[~objeect_masks[1], :].numpy()
    return [positive_embeddings], [negative_embeddings]

def get_top_obj_pos_bottom_obj_neg(latent_state, obj_df, objeect_masks):
    """
    Top object is positive, bottom object is negative. 
    No label for background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    if obj_df.iloc[0]['Center (x, y)'][1] < obj_df.iloc[1]['Center (x, y)'][1]:
        positive_embeddings = latent_state[objeect_masks[0], :].numpy()
        negative_embeddings = latent_state[objeect_masks[1], :].numpy()
    else:
        positive_embeddings = latent_state[objeect_masks[1], :].numpy()
        negative_embeddings = latent_state[objeect_masks[0], :].numpy()
    return [positive_embeddings], [negative_embeddings]


def get_top_obj_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Top object is positive, others are negative including background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    if obj_df.iloc[0]['Center (x, y)'][1] < obj_df.iloc[1]['Center (x, y)'][1]:
        positive_embeddings = latent_state[objeect_masks[0], :].numpy()
        negative_embeddings = latent_state[~objeect_masks[0], :].numpy()
    else:
        positive_embeddings = latent_state[objeect_masks[1], :].numpy()
        negative_embeddings = latent_state[~objeect_masks[1], :].numpy()
    return [positive_embeddings], [negative_embeddings]


def get_bottom_obj_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Bottom object is positive, others are negative including background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    if obj_df.iloc[0]['Center (x, y)'][1] > obj_df.iloc[1]['Center (x, y)'][1]:
        positive_embeddings = latent_state[objeect_masks[0], :].numpy()
        negative_embeddings = latent_state[~objeect_masks[0], :].numpy()
    else:
        positive_embeddings = latent_state[objeect_masks[1], :].numpy()
        negative_embeddings = latent_state[~objeect_masks[1], :].numpy()
    return [positive_embeddings], [negative_embeddings]


def get_triangle_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Top object is positive, others are negative including background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    positive_token_mask = np.zeros_like(objeect_masks[0], dtype=bool)
    for i in range(len(obj_df)):
        if obj_df.iloc[i]['Shape'] == "Triangle":
            positive_token_mask = positive_token_mask | objeect_masks[i]

    positive_embeddings = [latent_state[positive_token_mask, :].numpy()]
    negative_embeddings = [latent_state[~positive_token_mask, :].numpy()]
    return positive_embeddings, negative_embeddings


def get_circle_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Top object is positive, others are negative including background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    positive_token_mask = np.zeros_like(objeect_masks[0], dtype=bool)
    for i in range(len(obj_df)):
        if obj_df.iloc[i]['Shape'] == "Circle":
            positive_token_mask = positive_token_mask | objeect_masks[i]

    positive_embeddings = [latent_state[positive_token_mask, :].numpy()]
    negative_embeddings = [latent_state[~positive_token_mask, :].numpy()]
    return positive_embeddings, negative_embeddings


def get_square_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Top object is positive, others are negative including background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    positive_token_mask = np.zeros_like(objeect_masks[0], dtype=bool)
    for i in range(len(obj_df)):
        if obj_df.iloc[i]['Shape'] == "Square":
            positive_token_mask = positive_token_mask | objeect_masks[i]

    positive_embeddings = [latent_state[positive_token_mask, :].numpy()]
    negative_embeddings = [latent_state[~positive_token_mask, :].numpy()]
    return positive_embeddings, negative_embeddings


def get_red_triangle_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Top object is positive, others are negative including background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    positive_token_mask = np.zeros_like(objeect_masks[0], dtype=bool)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if obj_df.iloc[i]['Shape'] == "Triangle" and Rvalue > 225 and Gvalue < 30 and Bvalue < 30:
            positive_token_mask = positive_token_mask | objeect_masks[i]
            
    positive_embeddings = [latent_state[positive_token_mask, :].numpy()]
    negative_embeddings = [latent_state[~positive_token_mask, :].numpy()]
    return positive_embeddings, negative_embeddings


def get_red_square_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Red squares are positive, others are negative including background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    positive_token_mask = np.zeros_like(objeect_masks[0], dtype=bool)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if obj_df.iloc[i]['Shape'] == "Square" and Rvalue > 225 and Gvalue < 30 and Bvalue < 30:
            positive_token_mask = positive_token_mask | objeect_masks[i]
            
    positive_embeddings = [latent_state[positive_token_mask, :].numpy()]
    negative_embeddings = [latent_state[~positive_token_mask, :].numpy()]
    return positive_embeddings, negative_embeddings


def get_red_circle_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Red circles are positive, others are negative including background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    positive_token_mask = np.zeros_like(objeect_masks[0], dtype=bool)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if obj_df.iloc[i]['Shape'] == "Circle" and Rvalue > 225 and Gvalue < 30 and Bvalue < 30:
            positive_token_mask = positive_token_mask | objeect_masks[i]
            
    positive_embeddings = [latent_state[positive_token_mask, :].numpy()]
    negative_embeddings = [latent_state[~positive_token_mask, :].numpy()]
    return positive_embeddings, negative_embeddings


def get_blue_triangle_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Blue triangles are positive, others are negative including background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    positive_token_mask = np.zeros_like(objeect_masks[0], dtype=bool)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if obj_df.iloc[i]['Shape'] == "Triangle" and Rvalue < 30 and Gvalue < 30 and Bvalue > 225:
            positive_token_mask = positive_token_mask | objeect_masks[i]
            
    positive_embeddings = [latent_state[positive_token_mask, :].numpy()]
    negative_embeddings = [latent_state[~positive_token_mask, :].numpy()]
    return positive_embeddings, negative_embeddings


def get_blue_square_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Blue squares are positive, others are negative including background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    positive_token_mask = np.zeros_like(objeect_masks[0], dtype=bool)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if obj_df.iloc[i]['Shape'] == "Square" and Rvalue < 30 and Gvalue < 30 and Bvalue > 225:
            positive_token_mask = positive_token_mask | objeect_masks[i]
            
    positive_embeddings = [latent_state[positive_token_mask, :].numpy()]
    negative_embeddings = [latent_state[~positive_token_mask, :].numpy()]
    return positive_embeddings, negative_embeddings


def get_blue_circle_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Blue circles are positive, others are negative including background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    positive_token_mask = np.zeros_like(objeect_masks[0], dtype=bool)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if obj_df.iloc[i]['Shape'] == "Circle" and Rvalue < 30 and Gvalue < 30 and Bvalue > 225:
            positive_token_mask = positive_token_mask | objeect_masks[i]
            
    positive_embeddings = [latent_state[positive_token_mask, :].numpy()]
    negative_embeddings = [latent_state[~positive_token_mask, :].numpy()]
    return positive_embeddings, negative_embeddings


def get_red_obj_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Red objects are positive, others are negative including background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    positive_token_mask = np.zeros_like(objeect_masks[0], dtype=bool)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if Rvalue > 225 and Gvalue < 30 and Bvalue < 30:
            positive_token_mask = positive_token_mask | objeect_masks[i]
            
    positive_embeddings = [latent_state[positive_token_mask, :].numpy()]
    negative_embeddings = [latent_state[~positive_token_mask, :].numpy()]
    return positive_embeddings, negative_embeddings


def get_blue_obj_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Blue object is positive, others are negative including background. 
    """
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    positive_token_mask = np.zeros_like(objeect_masks[0], dtype=bool)
    for i in range(len(obj_df)):
        Rvalue, Gvalue, Bvalue = obj_df.iloc[i]['Color (RGB)']
        if Rvalue < 30 and Gvalue < 30 and Bvalue > 225:
            positive_token_mask = positive_token_mask | objeect_masks[i]
            
    positive_embeddings = [latent_state[positive_token_mask, :].numpy()]
    negative_embeddings = [latent_state[~positive_token_mask, :].numpy()]
    return positive_embeddings, negative_embeddings


def get_obj_pos_others_neg(latent_state, obj_df, objeect_masks):
    """
    Top object is positive, others are negative including background. 
    """
    
    if len(obj_df) != 2:
        return np.empty([0, latent_state.shape[-1]]), np.empty([0, latent_state.shape[-1]])
    
    positive_token_mask = np.zeros_like(objeect_masks[0], dtype=bool)
    for i in range(len(obj_df)):
        positive_token_mask = positive_token_mask | objeect_masks[i]
            
    positive_embeddings = [latent_state[positive_token_mask, :].numpy()]
    negative_embeddings = [latent_state[~positive_token_mask, :].numpy()]
    return positive_embeddings, negative_embeddings



# %%
# try to classify top (+) and down (-) objects
def collect_pos_neg_embeddings(saveroot, t_index, prompt_ids, seed_ids=range(10), diffusion_pass=("cond",),
                               get_pos_neg_embeddings_func=get_top_obj_pos_others_neg):
    positive_embeddings = [] 
    negative_embeddings = []
    # 0,1,8,9 are the prompts that have above or below.
    for prompt_idx in tqdm(prompt_ids):
        for seed_idx in seed_ids:
            latent_file = f"red_blue_8_pos_rndembposemb_img_latent_residual_prompt{prompt_idx}_seed{seed_idx}.pkl"
            latent_path = os.path.join(saveroot, latent_file)
            data = pickle.load(open(latent_path, 'rb'))
            # data = torch.load(latent_path, map_location=torch.device('cpu'))
            image_logs = data['image_logs']
            batch_size = len(image_logs[0]['images'])
            residual_state_traj = data['block_11_residual_spatial_state_traj']
            for image_idx in range(batch_size):
                obj_df, obj_masks = find_classify_object_masks(image_logs[0]['images'][image_idx])
                if len(obj_df) != 2:
                    continue
                obj_masks_resized = [cv2.resize(obj_mask, (8, 8)) for obj_mask in obj_masks]
                obj_masks_resized_binary = [obj_mask > positive_threshold for obj_mask in obj_masks_resized]
                for which_pass in diffusion_pass:
                    if which_pass == "cond":
                        pos_embeddings, neg_embeddings = get_pos_neg_embeddings_func(residual_state_traj[t_index, batch_size + image_idx], 
                                                                                        obj_df, obj_masks_resized_binary)
                    elif which_pass == "uncond":
                        pos_embeddings, neg_embeddings = get_pos_neg_embeddings_func(residual_state_traj[t_index, image_idx], 
                                                                                        obj_df, obj_masks_resized_binary)
                    else:
                        raise ValueError(f"Invalid diffusion pass: {which_pass} (should be in ['cond', 'uncond'])")
                    positive_embeddings.extend(pos_embeddings)
                    negative_embeddings.extend(neg_embeddings)  

    positive_embeddings = np.vstack(positive_embeddings)
    negative_embeddings = np.vstack(negative_embeddings)
    return positive_embeddings, negative_embeddings


def collect_pos_neg_embeddings_layerwise(saveroot, t_index, prompt_ids, layer_id=11, seed_ids=range(10), diffusion_pass=("cond",),
                               get_pos_neg_embeddings_func=get_top_obj_pos_others_neg):
    positive_embeddings = [] 
    negative_embeddings = []
    # 0,1,8,9 are the prompts that have above or below.
    for prompt_idx in tqdm(prompt_ids):
        for seed_idx in seed_ids:
            latent_file = f"red_blue_8_pos_rndembposemb_img_latent_residual_allblocks_prompt{prompt_idx}_seed{seed_idx}.pkl"
            latent_path = os.path.join(saveroot, latent_file)
            data = pickle.load(open(latent_path, 'rb'))
            # data = torch.load(latent_path, map_location=torch.device('cpu'))
            image_logs = data['image_logs']
            batch_size = len(image_logs[0]['images'])
            residual_state_traj = data[f'block_{layer_id}_residual_spatial_state_traj']
            for image_idx in range(batch_size):
                obj_df, obj_masks = find_classify_object_masks(image_logs[0]['images'][image_idx])
                if len(obj_df) != 2:
                    continue
                obj_masks_resized = [cv2.resize(obj_mask, (8, 8)) for obj_mask in obj_masks]
                obj_masks_resized_binary = [obj_mask > positive_threshold for obj_mask in obj_masks_resized]
                for which_pass in diffusion_pass:
                    if which_pass == "cond":
                        pos_embeddings, neg_embeddings = get_pos_neg_embeddings_func(residual_state_traj[t_index, batch_size + image_idx], 
                                                                                        obj_df, obj_masks_resized_binary)
                    elif which_pass == "uncond":
                        pos_embeddings, neg_embeddings = get_pos_neg_embeddings_func(residual_state_traj[t_index, image_idx], 
                                                                                        obj_df, obj_masks_resized_binary)
                    else:
                        raise ValueError(f"Invalid diffusion pass: {which_pass} (should be in ['cond', 'uncond'])")
                    positive_embeddings.extend(pos_embeddings)
                    negative_embeddings.extend(neg_embeddings)  

    positive_embeddings = np.vstack(positive_embeddings)
    negative_embeddings = np.vstack(negative_embeddings)
    return positive_embeddings, negative_embeddings
# %% [markdown]
# ### Modularized functions for linear probe training
#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def train_classifier(positive_embeddings, negative_embeddings, test_size=0.2, random_state=42, fit_intercept=False, solver='lbfgs', max_iter=100):
    """
    Train a logistic regression classifier to classify positive and negative embeddings. 
    No intercept / bias is added. 
    """
    # Combine positive and negative samples
    X = np.vstack([positive_embeddings, negative_embeddings])
    # Create labels (1 for positive, 0 for negative)
    y = np.concatenate([np.ones(len(positive_embeddings)), np.zeros(len(negative_embeddings))])
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Train the classifier
    clf = LogisticRegression(random_state=random_state, fit_intercept=fit_intercept, max_iter=max_iter, solver=solver, n_jobs=-1)
    clf.fit(X_train, y_train)
    # Evaluate performance
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    # compute the confusion matrix
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix: \n", cm)
    # print(f"Precision (TP / (TP + FP)): {cm[1,1] / (cm[1,1] + cm[1,0]):.3f}") # Bug fixed Dec27, 2024 cm[1,1] / (cm[1,1] + cm[0,1])
    # print(f"Recall (TP / (TP + FN)): {cm[1,1] / (cm[1,1] + cm[0,1]):.3f}")
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Precision (TP / (TP + FP)): {precision:.3f}")
    print(f"Recall (TP / (TP + FN)): {recall:.3f}")
    print(f"F1 score: {f1:.3f}")
    eval_dict = {
        "classifier": clf, 
        "train_score": train_score,
        "test_score": test_score,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    return clf, X, y, eval_dict


def get_projection_basis(clf):
    # Get the decision boundary normal vector (feature importance)
    feature_importance = clf.coef_[0]
    # Get the normalized decision boundary vector
    boundary_vector = feature_importance / np.linalg.norm(feature_importance)
    # Get an orthogonal vector for the second dimension
    basis_vector = np.zeros_like(boundary_vector)
    basis_vector[0] = 1
    orthogonal = basis_vector - (basis_vector @ boundary_vector) * boundary_vector
    orthogonal = orthogonal / np.linalg.norm(orthogonal)
    return boundary_vector, orthogonal


def project_data(X, boundary_vector, orthogonal):
    # Project data onto boundary vector and orthogonal vector
    X_boundary = X @ boundary_vector
    X_orthogonal = X @ orthogonal
    return np.column_stack([X_boundary, X_orthogonal])


def plot_classification(X_proj, y, title="Classification Visualization using Decision Boundary", s=9, alpha=0.3):
    # Split into positive and negative samples
    X_pos = X_proj[y==1]
    X_neg = X_proj[y==0]

    # Create scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(X_pos[:, 0], X_pos[:, 1], s=s, c='blue', label='Top object', alpha=alpha)
    plt.scatter(X_neg[:, 0], X_neg[:, 1], s=s, c='red', label='Bottom object', alpha=alpha) 
    # Plot decision boundary (vertical line at x=0 since boundary_vector is one axis)
    plt.axvline(x=0, color='k', linestyle='-')
    
    plt.xlabel('Projection onto Decision Boundary Normal')
    plt.ylabel('Orthogonal Direction')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def train_classifier_and_visualize(positive_embeddings, negative_embeddings, visualize=True, s=9, alpha=0.3, solver='lbfgs', max_iter=100):
    # Train classifier and get data
    clf, X, y, eval_dict = train_classifier(positive_embeddings, negative_embeddings, solver=solver, max_iter=max_iter)
    # Get projection vectors
    boundary_vector, orthogonal = get_projection_basis(clf)
    if visualize:
        # Project the data
        X_proj = project_data(X, boundary_vector, orthogonal)
        # Visualize the results
        plot_classification(X_proj, y, s=s, alpha=alpha)
    return clf, boundary_vector, eval_dict


# %%
def visualize_vecprod_activation_heatmap(boundary_vector, saveroot, t_index, prompt_idx, seed_idx=0, use_relu=True, title_str=""):
    latent_file = f"red_blue_8_pos_rndembposemb_img_latent_residual_prompt{prompt_idx}_seed{seed_idx}.pkl"
    latent_path = os.path.join(saveroot, latent_file)
    with open(latent_path, 'rb') as f:
        data = pickle.load(f)
    prompt = data['prompt']
    image_logs = data['image_logs']
    batch_size = len(image_logs[0]['images'])
    residual_state_traj = data['block_11_residual_spatial_state_traj']
    if use_relu:
        activation = torch.relu(residual_state_traj @ boundary_vector) # (t steps, batch size, 8, 8)
    else:
        activation = (residual_state_traj @ boundary_vector) # (t steps, batch size, 8, 8)
    figh, axs = plt.subplots(5, 15, figsize=(35, 15))
    axs = axs.flatten()
    for i in range(batch_size):
        axs[3*i].imshow(image_logs[0]['images'][i])
        axs[3*i].axis('off')
        axs[3*i+1].imshow(activation[t_index, i].reshape(8, 8))
        axs[3*i+1].axis('off')
        axs[3*i+1].set_title(f"uncond")
        axs[3*i+2].imshow(activation[t_index, batch_size + i].reshape(8, 8))
        axs[3*i+2].axis('off')
        axs[3*i+2].set_title(f"cond")
    plt.suptitle(f"{prompt} t index={t_index} seed={seed_idx}, {'Relu' if use_relu else ''} dot prod activation {title_str}", fontsize=18)
    plt.tight_layout()
    plt.show()
    return figh


# %%
positive_threshold = 180 
saveroot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_rndembdposemb_DiT_B_pilot/latent_store_norm2"
figdir = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt/results/objrel_rndembdposemb_DiT_B_pilot/feature_classifier/norm_2'
# %%
# seed = 0
# for t_index in reversed(range(14)):
#     for get_pos_neg_embeddings_func, annot_label in [
#                         (get_red_triangle_pos_others_neg, "red_triangle_vs_others"),
#                         (get_obj_pos_others_neg, "obj_vs_others"),
#                         (get_red_obj_pos_others_neg, "red_obj_vs_others"),
#                         (get_triangle_pos_others_neg, "triangle_vs_others"), 
#                         (get_circle_pos_others_neg, "circle_vs_others"),
#                         # (get_top_obj_pos_others_neg, "topobj_vs_others"),
#                         # (get_top_obj_pos_bottom_obj_neg, "topobj_vs_bottomobj"),
#                                                     ]:
#         for training_pass in [("cond",), ("uncond",), ("cond", "uncond")]:
#             training_pass_str = "-".join(training_pass)+"Pass"
#             positive_embeddings, negative_embeddings = collect_pos_neg_embeddings(saveroot, t_index=t_index, prompt_ids=range(16), seed_ids=range(10), 
#                                                                                 get_pos_neg_embeddings_func=get_pos_neg_embeddings_func, diffusion_pass=training_pass)
#             clf, boundary_vector, eval_dict = train_classifier_and_visualize(positive_embeddings, negative_embeddings, visualize=False, solver='liblinear')
#             pkl.dump({"classifier": clf, "boundary_vector": boundary_vector, "eval_dict": eval_dict}, 
#                     open(join(figdir, f"red_blue_8_relation_diff_t{t_index}_{annot_label}_classifier_{training_pass_str}.pkl"), 'wb'))
#             for prompt_idx in range(16):
#                 figh = visualize_vecprod_activation_heatmap(boundary_vector, saveroot, t_index=t_index, prompt_idx=prompt_idx, seed_idx=seed, use_relu=True, title_str=f"Claissifier {annot_label} {training_pass_str} pass")
#                 saveallforms(figdir, f"red_blue_8_relation_diff_t{t_index}_prompt{prompt_idx}_seed{seed}_{annot_label}_relu_heatmap_{training_pass_str}", figh)
#                 figh = visualize_vecprod_activation_heatmap(boundary_vector, saveroot, t_index=t_index, prompt_idx=prompt_idx, seed_idx=seed, use_relu=False, title_str=f"Claissifier {annot_label} {training_pass_str} pass")
#                 saveallforms(figdir, f"red_blue_8_relation_diff_t{t_index}_prompt{prompt_idx}_seed{seed}_{annot_label}_linear_heatmap_{training_pass_str}", figh)
#                 plt.close("all")


#%%
# vis_seed = 0
import argparse

parser = argparse.ArgumentParser(description='Train classifiers on latent features')
parser.add_argument('--layer_id', type=int, default=11,
                    help='Layer ID to analyze (default: 11)')
parser.add_argument('--t_start', type=int, default=0,
                    help='Starting timestep index (default: 0)')
parser.add_argument('--t_end', type=int, default=14,
                    help='Ending timestep index, not included (default: 14)')

args = parser.parse_args()

layer_id = args.layer_id
for t_index in reversed(range(args.t_start, args.t_end)):
    for get_pos_neg_embeddings_func, annot_label in [
                        (get_left_obj_pos_right_obj_neg, "leftobj_vs_rightobj"),
                        #(get_right_obj_pos_left_obj_neg, "rightobj_vs_leftobj"),
                        (get_left_obj_pos_others_neg, "leftobj_vs_others"),
                        (get_right_obj_pos_others_neg, "rightobj_vs_others"),
                        # (get_obj_pos_others_neg, "obj_vs_others"),
                        # (get_top_obj_pos_others_neg, "topobj_vs_others"),
                        # (get_bottom_obj_pos_others_neg, "bottomobj_vs_others"), 
                        # (get_top_obj_pos_bottom_obj_neg, "topobj_vs_bottomobj"),
                        # (get_triangle_pos_others_neg, "triangle_vs_others"),
                        # (get_circle_pos_others_neg, "circle_vs_others"),
                        # (get_square_pos_others_neg, "square_vs_others"),
                        # (get_red_obj_pos_others_neg, "red_obj_vs_others"),
                        # (get_red_triangle_pos_others_neg, "red_triangle_vs_others"),
                        # (get_red_circle_pos_others_neg, "red_circle_vs_others"),
                        # (get_red_square_pos_others_neg, "red_square_vs_others"),
                        # (get_blue_obj_pos_others_neg, "blue_obj_vs_others"),
                        # (get_blue_triangle_pos_others_neg, "blue_triangle_vs_others"),
                        # (get_blue_circle_pos_others_neg, "blue_circle_vs_others"),
                        # (get_blue_square_pos_others_neg, "blue_square_vs_others"),
                            ]:
        for training_pass in [("cond",), ("uncond",), ("cond", "uncond")]:
            training_pass_str = "-".join(training_pass)+"Pass"
            savepath = join(figdir, f"red_blue_8_relation_diff_layer{layer_id}_t{t_index}_{annot_label}_classifier_{training_pass_str}.pkl")
            if os.path.exists(savepath):
                print(f"Skipping {savepath} because it already exists")
                continue
            positive_embeddings, negative_embeddings = collect_pos_neg_embeddings_layerwise(saveroot, t_index=t_index, prompt_ids=range(64), layer_id=layer_id, seed_ids=np.random.choice(10, size=3, replace=False),
                                                                                get_pos_neg_embeddings_func=get_pos_neg_embeddings_func, diffusion_pass=training_pass)
            clf, boundary_vector, eval_dict = train_classifier_and_visualize(positive_embeddings, negative_embeddings,
                                                    visualize=False, solver='liblinear', max_iter=100) # liblinear | lbfgs | saga
            pkl.dump({"classifier": clf, "boundary_vector": boundary_vector, "eval_dict": eval_dict},
                    open(savepath, 'wb'))
                # for prompt_idx in range(16):
                #     figh = visualize_vecprod_activation_heatmap(boundary_vector, saveroot, t_index=t_index, prompt_idx=prompt_idx, seed_idx=vis_seed, use_relu=True, title_str=f"Claissifier {annot_label} {training_pass_str} pass")
                #     saveallforms(figdir, f"red_blue_8_relation_diff_layer{layer_id}_t{t_index}_prompt{prompt_idx}_seed{vis_seed}_{annot_label}_relu_heatmap_{training_pass_str}", figh)
                #     figh = visualize_vecprod_activation_heatmap(boundary_vector, saveroot, t_index=t_index, prompt_idx=prompt_idx, seed_idx=vis_seed, use_relu=False, title_str=f"Claissifier {annot_label} {training_pass_str} pass")
                #     saveallforms(figdir, f"red_blue_8_relation_diff_layer{layer_id}_t{t_index}_prompt{prompt_idx}_seed{vis_seed}_{annot_label}_linear_heatmap_{training_pass_str}", figh)
                #     plt.close("all")
