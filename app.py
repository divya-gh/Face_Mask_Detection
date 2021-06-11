#Import necessary libraries
from flask import Flask, render_template, Response, jsonify
import cv2
import get_data as gd
#Initialize the Flask app
app = Flask(__name__)


#for local webcam use cv2.VideoCapture(0)

#camera = cv2.VideoCapture(0)




@app.route('/')
def index():
    return render_template('index.html' )


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