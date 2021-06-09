#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
#Initialize the Flask app
app = Flask(__name__)


#for local webcam use cv2.VideoCapture(0)

#camera = cv2.VideoCapture(0)




@app.route('/')
def index():
    return render_template('index.html')


@app.route("/get_image/<img>")
def get_predicted_image(img):
    
    return redirect("/", code=302)













if __name__ == "__main__":
    app.run(debug=True)