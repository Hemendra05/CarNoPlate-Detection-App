import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import numpy as np
import matplotlib.pyplot as plt
import cv2
import boto3
import requests
import xmltodict
import json



# UPLOAD_FOLDER = ""
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Connectig to AWS S3
s3 = boto3.resource('s3')
bucket = "hemendra-test-bucket"

#Connecting to AWS Rekognition service
rekognition = boto3.client('rekognition')


def extractLicenceNo(image):

    carPlateImg = cv2.imread(image)
    carImgRBG = cv2.cvtColor(carPlateImg, cv2.COLOR_BGR2RGB)

    licensePlateModel = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    licenceDetails = []

    # Create function to crop only the car plate region itself
    car = licensePlateModel.detectMultiScale(image,scaleFactor=1.1, minNeighbors=3)
    for licencePlate in car:
        x=licencePlate[0]
        y=licencePlate[1]
        w=licencePlate[2]
        h=licencePlate[3]
        licensePlate = image[y:y+h ,x:x+w] # Adjusted to extract specific region of interest i.e. car license plate
        # Enlarge image for further processing later on
        width = int(licensePlate.shape[1] * 2)
        height = int(licensePlate.shape[0] * 2)
        dim = (width, height)
        resizedLicencePlate = cv2.resize(licensePlate, dim, interpolation = cv2.INTER_AREA)
        # Savinf Resized Licence Plate locally
        cv2.imwrite("licensePlate.jpg", resizedLicencePlate)
        # Uploading Licence Plate to WS S3
        s3.Bucket(bucket).upload_file("licensePlate.jpg", "licensePlate.jpg")
        # Detecting No plate text using AWS Rekognition
        response = rekognition.detect_text(
        Image={
            'S3Object': {
                'Bucket': 'hemendra-test-bucket',
                'Name': 'licensePlate.jpg',
                }
            }
        )

        licensePlateNo = response['TextDetections'][0]['DetectedText']

        vehicleInfo = requests.get("http://www.regcheck.org.uk/api/reg.asmx/CheckIndia?RegistrationNumber={}&username=hemendra005".format(licencePlateNo))
        data = xmltodict.parse(vehicleInfo.content)
        jdata = json.dumps(data)
        df = json.loads(jdata)
        df1 = json.loads(df['Vehicle']['vehicleJson'])
        licenceDetails.append(df1)

        licenceDetails.append(licensePlateNo)
    return licenceDetails



app = Flask("myapp")
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/index/<id>')
def dfs(id):
    return id

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/uploadFile', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(filename)
            return "Saved"
    return "Successful"







app.run(host='0.0.0.0', port=5000, debug=True)
