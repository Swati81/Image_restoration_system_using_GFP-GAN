from flask import Flask,render_template,request,send_file
from flask_cors import cross_origin
from zipfile import ZipFile
import cv2
import os
from decode import Decode
from inference_gfpgan import restoration

app = Flask(__name__)


@app.route('/',methods=['GET'])
@cross_origin()
def index():
    return render_template('index.html')

@app.route('/result',methods=['GET','POST'])
@cross_origin()
def result():
    if request.method == 'POST':
        image = request.json['image']
        img = Decode(image).copy()
        cv2.imwrite('static/input.png',img)
        restoration()
        return render_template('index.html')
    return render_template('index.html')






if __name__=="__main__":
    app.run(host='0.0.0.0',port=8800)