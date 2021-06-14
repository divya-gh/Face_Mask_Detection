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
    # Get files from the folder
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
def predicted_image(img_file ="people1.jpg"):
    #print("img_file", img_file)
    experiment_images = ["people1.jpg", "people2.jpg", "people3.jpg", "people4.jpg", "people5.jpg", "people6.jpg", "people7.jpg"]
    if img_file not in experiment_images:
        base_path = "./Resources/UploadPic/"
    else:
        base_path = "./Resources/Experiment/"
    img_path = base_path +  img_file
    print("path:", img_path)
    data = tm.prediction(img_path)
    #print("data", data)
    return data




        
    

if __name__ == "__main__":
   
    images = get_sel_images()
    #print(images)
    img_file = predicted_image("people1.jpg")