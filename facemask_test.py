

import cv2
from tensorflow.keras.models import load_model
import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
import time
import imutils
import os
import time


face_model = cv2.CascadeClassifier('./Resources/haarcascade_frontalface_default.xml')
model = load_model("./Saved_Model/keras_model.hdf5")

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

                cv2.putText(frame,mask_label[round(mask_result[0][0])],(x, y-10), cv2.FONT_HERSHEY_SIMPLEX,2,color_label[round(mask_result[0][0])],3)
                cv2.rectangle(frame,(x,y),(x+w,y+h), color_label[round(mask_result[0][0])],3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

  
        # cv2.imshow("Frame", frame)

    video_capture.release()
    cv2.destroyAllWindows()

    # return round(pred[0][0]*100)

if __name__ == "__main__":
    livePrediction()





