#Import necessary libraries
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
import cv2
import os
import imghdr
import get_data as gd
from facemask_test import livePrediction


#Initialize the Flask app
app = Flask(__name__)


#for local webcam use cv2.VideoCapture(0)

#camera = cv2.VideoCapture(0)

# --------- TODO Copy to Main ------------- #
UPLOAD_FOLDER = './static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0) 
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')
# --------- Copy to Main ------------- #

@app.route('/')
@app.route('/<task>')
def index(task=""):
    return render_template('index.html' ,task=task)

# --------- TODO Copy to Main ------------- #
@app.route('/upload', methods=["GET", "POST"])
def uploader():
   if request.method == 'POST':
      f = request.files['file']
      filename = secure_filename(f.filename)
      if filename != '':
          file_ext = os.path.splitext(filename)[1]
          if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(f.stream):
              abort(400)
          f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
          print("Yay! It Worked!")
          return '', 204
# --------- Copy to Main ------------- #

@app.route("/get_image/<img>")
def get_predicted_image(img):
    print("image file rendered:",img)
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        cap.release()
        cv2.destroyAllWindows()
        print("destroyed")    
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