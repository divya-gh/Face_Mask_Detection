#Import necessary libraries
from flask import Flask, render_template, Response, jsonify
import cv2
import get_data as gd
from facemask_test import livePrediction


#Initialize the Flask app
app = Flask(__name__)


#for local webcam use cv2.VideoCapture(0)

#camera = cv2.VideoCapture(0)




@app.route('/')
@app.route('/<task>')
def index(task=""):
    return render_template('index.html' ,task=task)


@app.route("/get_image/<img>")
def get_predicted_image(img):
    print("image file rendered:",img)
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        cap.release()
        cv2.destroyAllWindows()
        print("distroyed")
    data = gd.predicted_image(img)
    print(jsonify(data))
    return jsonify(data)

@app.route("/api/v1.0/select_option")
def get_selected_images():    
    img_files = gd.get_sel_images()
    return jsonify(img_files)



@app.route("/video_feed")
def video_feed():    
    return Response(livePrediction(), mimetype='multipart/x-mixed-replace; boundary=frame')















if __name__ == "__main__":
    app.run(debug=True)