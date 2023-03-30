import tensorflow as tf
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
class neuronal_network:
    def __init__(self):
        self.ruta_dataset = os.path.dirname(os.path.abspath(__file__))+'/data/train/'
        self.classification = os.listdir(self.ruta_dataset)
        self.images = []
        self.num_layers = 0
        self.num_filters = 0
        self.num_classes = 0
        self.learning_rate = 0.005
        self.model = None
        self.labels = []
        self.history_list = []
        self.accuracy_list = []

    def preprocess_images(self,folds):
        x = 0
        for dirr in self.classification:
            for image in os.listdir(self.ruta_dataset + dirr):
                img = Image.open(self.ruta_dataset + dirr + '/' + image).convert('RGB').resize((100,100),)
                img = np.asanyarray(img)
                self.images.append(img) 
                self.labels.append(x)
            x += 1  
            self.num_classes += 1   
            
        self.images = np.asanyarray(self.images)
        self.labels = np.asanyarray(self.labels)
        self.images = self.images / 255

        self.kfold = KFold(n_splits=folds, shuffle=True)
        
    def define_model(self, num_layers_dense, num_neurons, activation):
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(100, 100, 3), activation='relu'),)
        model.add(tf.keras.layers.MaxPooling2D(2, 2),)
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),)
        model.add(tf.keras.layers.MaxPooling2D(2, 2),)
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),)
        model.add(tf.keras.layers.MaxPooling2D(2, 2),)
        model.add(tf.keras.layers.Flatten())
        print(f"num_layers: {num_layers_dense} num_neurons: {num_neurons} f_act: {activation}")
        for i in range(num_layers_dense):
            model.add(tf.keras.layers.Dense(num_neurons[i], activation=activation[i]))
        
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        
        opt = tf.keras.optimizers.Adam(learning_rate=0.005)

        model.compile(
            optimizer = opt,
            loss = tf.losses.SparseCategoricalCrossentropy(),
            metrics = ['accuracy']
        )        
        
        self.model = model
        
    def train_model(self, epochs):

        for fold, (train_index, val_index) in enumerate(self.kfold.split(self.images, self.labels)):
            # print(f'Train model for {fold} fold...')
            x_train, y_train = self.images[train_index], self.labels[train_index]
            x_val, y_val = self.images[val_index], self.labels[val_index]

            history = self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val),verbose=0)
            loss, accurancy = self.model.evaluate(x_val, y_val, verbose=0)

            self.history_list.append(history)
        self.accurancy = accurancy*100
        print("accurancy: ",accurancy*100)
            
    def save_model(self):
        target_dir = './model/'
        
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
            
        self.model.save('./model/model.h5')
        self.model.save_weights('./model/weights.h5')
        
    def evaluate(self, num_layers_dense, num_neurons, activation):
        self.preprocess_images(folds=5)
        self.define_model(num_layers_dense, num_neurons, activation)
        self.train_model(20)
        self.save_model()

# nn = neuronal_network()
# nn.evaluate(1,[100],['relu'])