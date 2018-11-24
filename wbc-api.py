import os, io, cv2
import numpy as np
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from scipy.misc import imresize
from wbcbinaryclassification import train_data, predict_data, toArray

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'people_photo')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        print('No selected file')
        return redirect(request.url)
    elif file and allowed_file(file.filename):
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img = cv2.imdecode(data, color_image_flag)
    
        imageArray = imresize(img, (120, 160, 3))
        imageArray = toArray([toArray(imageArray)])
    
        predict = predict_data(imageArray)
        
        if round(predict) != 1:
            return "MONONUCLEAR"
        else:
            return "POLYNUCLEAR"
        

@app.route('/train', methods=['GET'])
def predict():
    train_data()
    return "Done"

if __name__ == '__main__':
    app.run(debug=True)
