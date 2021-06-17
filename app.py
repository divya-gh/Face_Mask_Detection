#Import necessary libraries
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, abort
import cv2
from werkzeug.utils import secure_filename
import imghdr
from get_data import prediction, get_sel_images
from get_data import livePrediction
import os


#Initialize the Flask app
app = Flask(__name__)


# ---------------------prediction function---------------------------#
# predict  function   
def predicted_image(img_file):
    #print("img_file", img_file)
    experiment_images = ["people1.jpg", "people2.jpg", "people3.jpg", "people4.jpg", "people5.jpg", "people6.jpg", "people7.jpg"]
    if img_file not in experiment_images:
    # --------- TODO Copy to Main ------------- #
        # base_path = "./static/upload/"
    # --------- Copy to Main ------------- #
        base_path = "./Resources/UploadPic/"
    else:
        base_path = "./Resources/Experiment/"
    img_path = base_path + img_file
    print("path:", img_path)
    data = prediction(img_path)
    #print("data", data)
    return data
# -------------------------------------------------------------------#


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


# default app route
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

# app route for image prediction
@app.route("/get_image/<img>")
def get_predicted_image(img):
    print("image file rendered:",img)
    #img = "people5.jpg"
    data = predicted_image(img)   
    print(data)
    return jsonify(data)

# app route for image file selection
@app.route("/api/v1.0/select_option")
def get_selected_images():    
    img_files = get_sel_images()
    return jsonify(img_files)


# app route for video feed
@app.route("/video_feed")
def video_feed():    
    return Response(livePrediction(), mimetype='multipart/x-mixed-replace; boundary=frame')















if __name__ == "__main__":
    app.run(debug=True)