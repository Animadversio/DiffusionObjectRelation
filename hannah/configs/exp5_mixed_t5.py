# Experiment 5: Mixed dataset with T5 embeddings
num_images = 10000
resolution = 128
radius = 16
single_ratio = 0.3
model_max_length = 20
dataset_type = "Mixed"
encoder_type = "T5"
dataset_name = "objectRelMixed_pilot1_T5"
pixart_dir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/PixArt"
save_dir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/Diffusion_ObjectRelation"
using_existing_img_txt = True

validation_prompts =  [
        "triangle",
        "square",
        "circle",
        "red",
        "blue",
        "red square",
        "blue circle",
        "blue triangle",
        "a red square",
        "a blue circle",
        "a blue triangle",
        "the blue square",
        "the red circle",
        "the triangle",
        "the square",
        "the circle",
        "the",
        "or",
        "an",
        "red",
        "blue",
        "the red square above the blue circle",
        "blue triangle to the left of red square",
        "red circle below blue triangle",
        "red circle to the right of blue triangle"
        ] 