# Loading all images using imread from a given folder
import cv2
import os
from PIL import Image
import json
#clear Screen
os.system("cls")

import pandas as pd
import numpy as np
import Test_with_model as tm


# Function to load images from the folder
def load_images_from_folder(folder):
    img_files = []
    for filename in os.listdir(folder):
        img_files.append(filename)
    return img_files




# Get images from the experiments folder to render to selection option on launch page
def get_sel_images():
    # set the path
    path = "./Resources/Experiment/"

    # call the function load_images_from_folder to get images in an array
    images = load_images_from_folder(path)
    #print(images)
    return images

# predict  function   
def predicted_image(img_file = "people1.jpg"):
    print(img_file)
    base_path = "./Resources/Experiment/"
    img_path = base_path +  img_file
    print(img_path)
    data = tm.prediction(img_path)
    print(data)




        
    

if __name__ == "__main__":
   
    images = get_sel_images()
    #print(images)
    img_file = predicted_image("people5.jpg")