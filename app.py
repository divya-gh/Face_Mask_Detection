#Import necessary libraries
from flask import Flask, render_template, Response, jsonify
import cv2
# import imutils
# from genframes import gen_frames, capture_img
from facemask_test import livePrediction
import sched, time



import get_data as gd
#Initialize the Flask app
app = Flask(__name__)

## for coolbeans use application instead of app for the flask variable


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/video_feed")
def video_feed():
    # return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(livePrediction(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route("/get_image")
# def capture():
#     img = capture_img()

#     # img = cv2.imread('./Resources/Test/85-with-mask.jpg')

#     result = test_img(img)

#     return render_template('index2.html', result= result)

@app.route("/get_image/<img>")
def get_predicted_image(img):
    data = gd.predicted_image(img)
    print(data)
    return jsonify(data)

@app.route("/api/v1.0/select_option")
def get_selected_images():    
    img_files = gd.get_sel_images()
    return jsonify(img_files)

# @app.after_request
# def apply_caching(response):
#     response.headers["X-Frame-Options"] = "SAMEORIGIN"
#     return response












if __name__ == "__main__":
    app.run(debug=True)