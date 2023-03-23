import tensorflow as tf
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
#hacer prediccion
# from prediction import prediction_image
class neuronal_network:
    def __init__(self):
        self.ruta_dataset = os.path.dirname(os.path.abspath(__file__))+'/dataset/train/'
        self.classification = os.listdir(self.ruta_dataset)
        self.images = []
        self.num_layers = 0
        self.num_filters = 0
        self.num_classes = 0
        self.learning_rate = 0
        self.model = None
        self.labels = []
        self.history_list = []
        self.accuracy_list = []
        self.num_folds = 0

    def preprocess_images(self):
        x = 0
        for dirr in self.classification:
            for image in os.listdir(self.ruta_dataset + dirr):
                img = Image.open(self.ruta_dataset + dirr + '/' + image).convert('RGB').resize((50,50),)
                img = np.asanyarray(img)
                self.images.append(img) 
                self.labels.append(x)
            x += 1  
            self.num_classes += 1   
            
        self.images = np.asanyarray(self.images)
        self.labels = np.asanyarray(self.labels)
        self.images = self.images / 255
        
    def define_model(self, num_layers_dense, num_neurons, activation):
        #Datos que pasa el individuo
        self.num_layers = 4
        self.num_filters = 16
        self.learning_rate = 0.005
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(self.num_filters, (3, 3), input_shape=(50, 50, 3), activation='relu'))
        
        for i in range(self.num_layers - 1):
            model.add(tf.keras.layers.Conv2D(self.num_filters, (3, 3), activation='relu'),)
            
        model.add(tf.keras.layers.MaxPooling2D(2, 2),)
        model.add(tf.keras.layers.Flatten())
        for i in range(num_layers_dense):
            model.add(tf.keras.layers.Dense(num_neurons[i], activation=activation[i]))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer = opt,
            loss = tf.losses.SparseCategoricalCrossentropy(),
            metrics = ['accuracy']
        )        
        self.model = model
        
    def train_model(self):
        self.accuracy_list = []
        self.num_folds = 3
        
        kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=7)

        for fold, (train_index, val_index) in enumerate(kfold.split(self.images, self.labels)):
            # print(f'Train model for {fold} fold...')
            x_train, y_train = self.images[train_index], self.labels[train_index]
            x_val, y_val = self.images[val_index], self.labels[val_index]

            history = self.model.fit(x_train, y_train, epochs=3, validation_data=(x_val, y_val), verbose=0)
            loss, accurancy = self.model.evaluate(x_val, y_val, verbose=0)

            self.history_list.append(history)
            self.accuracy_list.append(accurancy*100)
            
    def save_model(self):
        target_dir = './model/'
        
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
            
        self.model.save('./model/model.h5')
        self.model.save_weights('./model/weights.h5')
        
    def evaluate(self,num_layers_dense,num_neurons,activation):
        self.preprocess_images()
        self.define_model(num_layers_dense,num_neurons,activation)
        self.train_model()
        self.save_model()

# accurancy_trains = []
# num_trains = 2

# nn = neuronal_network()
# nn.preprocess_images()
# nn.define_model(
#         num_layers_dense=1,
#         num_neurons=[100],
#         activation=['relu']
#     )
# for i in range(num_trains):
#     print('Train: ',i)
#     nn.train_model()
#     accurancy_trains.append(np.mean(nn.accuracy_list))

# print('accurancy: ',accurancy_trains)
# print('accurancy mean: ',np.mean(accurancy_trains))

#hacer prediccion
# pi = prediction_image()
# pi.evaluate()