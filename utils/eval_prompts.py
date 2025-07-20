
import cv2_eval_utils
import 

class PromptDataset:
    def __init__(self):
        self.data = {
            "singular": [
                            "triangle", 
                            "square", 
                            "blue", 
                            "red", 
                            "above", 
                            "below", 
                            "left", 
                            "right"
                        ],
            "ordering": [
                            "blue triangle", 
                            "triangle blue",
                            "triangle above square", 
                            "triangle square above", 
                            "above triangle square"
                        ],
            "fillers": [
                            "the the the blue triangle",
                            "of the blue triangle the the the",
                            "red square",
                            "to or the red square the the"
                        ],
            "2_colors": [
                            "red_is_above_blue",
                            "red_is_to_the_left_of_red",
                        ],
            "relational":[
                            "triangle_is_above_and_to_the_right_of_square",
                            "triangle_is_to_the_left_of_square",
                            "triangle_is_to_the_left_of_triangle",
                            "triangle_is_to_the_upper_left_of_square",
                        ],
            "relational_2_colors": [
                            "blue_circle_is_above_and_to_the_right_of_blue_square"
                            "blue_circle_is_above_blue_square",
                            "blue_square_is_to_the_right_of_red_circle",
                            "blue_triangle_is_above_red_triangle",
                            "blue_triangle_is_to_the_upper_left_of_red_square",
                            "circle_is_below_red_square",
                            "red_circle_is_above_square",
                            "red_circle_is_to_the_left_of_blue_square",
                            "triangle_is_above_red_circle", 
            ]
        }
    def get_prompts_by_type(self, type):
        return self.data[type]
    
    def get_prompt_types(self):
        return self.data.keys()
    