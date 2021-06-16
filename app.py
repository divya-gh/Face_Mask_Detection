#Import necessary libraries
from flask import Flask, render_template, Response, jsonify
import cv2
import get_data as gd
from get_data import livePrediction


#Initialize the Flask app
app = Flask(__name__)


# default app route
@app.route('/')
@app.route('/<task>')
def index(task=""):
    return render_template('index.html' ,task=task)

# app route for image prediction
@app.route("/get_image/<img>")
def get_predicted_image(img):
    print("image file rendered:",img)
    data = gd.predicted_image(img)
    print(jsonify(data))
    return jsonify(data)

# app route for image file selection
@app.route("/api/v1.0/select_option")
def get_selected_images():    
    img_files = gd.get_sel_images()
    return jsonify(img_files)


# app route for video feed
@app.route("/video_feed")
def video_feed():    
    return Response(livePrediction(), mimetype='multipart/x-mixed-replace; boundary=frame')















if __name__ == "__main__":
    app.run(debug=True)