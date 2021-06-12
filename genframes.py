import cv2
camera = cv2.VideoCapture(0)

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

# def getFrame(sec, count):
#     
#     return hasFrames

# def capture_img():
#     sec = 0
#     # frameRate = 30 #//it will capture image in each 30 second interval
#     count=1
#     img_path = getFrame(sec, count)
    # camera.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
#     hasFrames,image = camera.read()
#     if hasFrames:
#         img_path = cv2.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
#       
#  #        while success:
#     #     count = count + 1
#     #     sec = sec + frameRate
#     #     sec = round(sec, 2)
#     #     success = getFrame(sec, count)

#     return img_path