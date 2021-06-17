# Loading all images using imread from a given folder
import cv2
import os
# from PIL import Image
import json
#clear Screen
os.system("cls")

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
import time
# import imutils


face_model = cv2.CascadeClassifier('./Resources/haarcascade_frontalface_default.xml')
model = load_model("./Model_Train_Test/keras_model.hdf5")

def prediction(path):

    # get Image from the path rendered
    img = cv2.imread(path)   
    # Make predictions on the testing set
    sample_img = cv2.resize(img,(128,128))
    sample_img = np.reshape(sample_img,[1,128,128,3])
    sample_img = sample_img/255.0
    pred = model.predict(sample_img)
    #print(pred)   

    # Uing Opencv2 to find social distancing and show mask prediction on the image
    mask_label = {0:'Mask Found',1:'No Mask Found'}
    color_label = {0:(0,255,0),1:(255,0,0)}
    MIN_DISTANCE = 0

    # convert Image to grayscale for object identification
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    faces = face_model.detectMultiScale(img,scaleFactor=1.2, minNeighbors=4)

    #check for no. of faces in the image
    # if more than one face found,
    if len(faces)>=1:
        label = [0 for i in range(len(faces))]

        # convert Image to color for rescaling and predicton
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image

        # For every face found, 
        predict_result =[]

        for i in range(len(faces)):
            (x,y,w,h) = faces[i]
            crop = new_img[y:y+h,x:x+w]
            crop = cv2.resize(crop,(128,128))
            crop = np.reshape(crop,[1,128,128,3])/255.0    
            mask_result = model.predict(crop)

            #print(color_label[round(mask_result[0][0])])
            predict_result.append(round(mask_result[0][0]))
            cv2.putText(new_img,mask_label[round(mask_result[0][0])],(x, y-10), cv2.FONT_HERSHEY_SIMPLEX,1.8,color_label[round(mask_result[0][0])],2)

            cv2.rectangle(new_img,(x,y),(x+w,y+h), color_label[round(mask_result[0][0])],3)

        plt.figure(figsize=(10,10))
        plt.imshow(new_img)
        predicted_path = "./static/Images/predicted_image.jpg"
        plt.savefig(predicted_path,bbox_inches='tight')

        # create a dictionary of results
        image_data = {
            "prediction" : predict_result,
            "Image_path" : predicted_path
        }

    else:
        print("No image")
        image_data = { "Prediction": "No Image"}


    return image_data

def livePrediction():
    print("Streaming started")
    video_capture = cv2.VideoCapture(0)

    mask_label = {0:'Mask Found',1:'No Mask Found'}
    color_label = {0:(0,255,0),1:(0,0,255)}
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        success, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_model.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    
        # convert the input frame from BGR to RGB 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #check for no. of faces in the image
        # if more than one face found, check if they are social distancing (extra code)
        if len(faces)>=1:
            label = [0 for i in range(len(faces))]
            
            # For every face found,   
            for i in range(len(faces)):
                (x,y,w,h) = faces[i]
                crop = rgb[y:y+h,x:x+w]
                crop = cv2.resize(crop,(128,128))
                crop = np.reshape(crop,[1,128,128,3])/255.0        
                mask_result = model.predict(crop)
                #print(color_label[round(mask_result[0][0])])

                cv2.putText(frame,mask_label[round(mask_result[0][0])],(x, y-10), cv2.FONT_HERSHEY_SIMPLEX,1,color_label[round(mask_result[0][0])],2)
                cv2.rectangle(frame,(x,y),(x+w,y+h), color_label[round(mask_result[0][0])],3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

  
        # cv2.imshow("Frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        key = cv2.waitKey(10)
    
        if key == 27: 
            break
    video_capture.release()
    cv2.destroyAllWindows()

    # return round(pred[0][0]*100)

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
# def predicted_image(img_file):
#     #print("img_file", img_file)
#     experiment_images = ["people1.jpg", "people2.jpg", "people3.jpg", "people4.jpg", "people5.jpg", "people6.jpg", "people7.jpg"]
#     if img_file not in experiment_images:
#         base_path = "./Resources/UploadPic/"
#     else:
#         base_path = "./Resources/Experiment/"
#     img_path = base_path + img_file
#     print("path:", img_path)
#     data = prediction(img_path)
#     #print("data", data)
#     return data




        
    

if __name__ == "__main__":
   
    images = get_sel_images()
    #print(images)
    img_file = predicted_image("people4.jpg")
    print(img_file)