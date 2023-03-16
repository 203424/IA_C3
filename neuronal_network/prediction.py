from keras.models import load_model
import os
from PIL import Image
import numpy as np

class prediction_image():
    def __init__(self):
        self.predictions_dir = os.listdir('data/prediction')
        self.predict_image = self.predictions_dir[0]
        self.predict_labels = os.listdir('./data/train/')
        self.weights = []
        self.history = []
        self.validate_image = []
        self.prediction_axis = []
        
    def import_model(self):
        self.model = './model/model.h5'
        self.weights = './model/weights.h5'
        self.history = load_model(self.model)
        self.history.load_weights(self.weights)
        
    def preprocess_image(self):
        img = Image.open('./data/prediction/' + self.predict_image).convert('RGB').resize((100,100),)
        img = np.asanyarray(img)
        self.validate_image.append(img)
        self.validate_image = np.asanyarray([img])
        
        print(self.validate_image.shape)
    
    def predict(self):
        predictions = self.history.predict(self.validate_image)
        
        for i in predictions:
            self.prediction_axis.append(self.predict_labels[np.argmax(i)])
        
        print('Predicción: ' + self.prediction_axis[0])
        
    def evaluate(self):
        self.import_model()
        self.preprocess_image()
        self.predict()
        
pi = prediction_image()
pi.evaluate()