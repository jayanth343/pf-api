from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from roboflow import Roboflow
import supervision as sv
import cv2
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
api = Api(app)

# Ensure the 'uploads' directory exists
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class FootPath(Resource):
    def post(self):
        try:
            if 'image' not in request.files:
                return jsonify({'Error': 'Image not received'})
            # print(request.files["electric"])
            image = request.files['image']
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            print(image_path)
            image.save(image_path)
            electric = request.form.get('electric') == 'true'  
            open_drain = request.form.get('openDrain') == 'true'
            electric = int(electric)
            open_drain = int(open_drain)
            print(f"Electric: {electric}, OpenDrain: {open_drain}")
            rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
            project = rf.workspace().project("orr")
            model = project.version(1).model

            result = model.predict(image_path, confidence=40)

            if hasattr(result, 'json'):
                result = result.json()

            detections = sv.Detections.from_inference(result)
            image = cv2.imread(image_path)
            masks = detections.mask
            if len(masks) > 0:
                totalPixels = sum(mask.size for mask in masks)
                footpathPixels = np.count_nonzero(masks[0]) 
                footpathPercentage = (((footpathPixels / totalPixels) * 100) - 5*electric - 3*open_drain)+10 if totalPixels else 0
            else:
                footpathPercentage = 0

            print(f"Prediction complete: {footpathPercentage}% footpath detected")

            return jsonify({'Percentage': footpathPercentage})

        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return jsonify({'Error': str(e)})


api.add_resource(FootPath, '/upload-image')

if __name__ == '__main__':
    app.run(debug=True)
