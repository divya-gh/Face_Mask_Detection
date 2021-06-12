#Import necessary libraries
from flask import Flask, render_template, Response, jsonify
import cv2
# import imutils
from genframes import gen_frames, capture_img
from facemask_test import test_img
import pickle


import get_data as gd
#Initialize the Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index2.html')

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/get_image")
def capture():
    img = capture_img()

    # img = cv2.imread('./Resources/Test/85-with-mask.jpg')

    result = test_img(img)

    return render_template('index2.html', result= result)

@app.route("/get_image/<img>")
def get_predicted_image(img):
    data = gd.predicted_image(img)
    print(data)
    return jsonify(data)

@app.route("/api/v1.0/select_option")
def get_selected_images():    
    img_files = gd.get_sel_images()
    return jsonify(img_files)














if __name__ == "__main__":
    app.run(debug=True)