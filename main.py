import numpy as np
import pandas as pd
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from PILasOPENCV import *

df = pd.read_csv('./Crop_details.csv')

Data = ImageDataGenerator(rescale=1 / 255, shear_range=0.2, horizontal_flip=True, vertical_flip=True)
Train_Data1 = Data.flow_from_directory('./kag2', target_size=(224, 224), batch_size=1)
Train_Data2 = Data.flow_from_directory('./crop_images', target_size=(224, 224), batch_size=1)
Test_Data = Data.flow_from_directory('./some_more_images', target_size=(224, 224), batch_size=1)

Model_Wrap = ResNet50(include_top=False, input_shape=(224, 224, 3))

for layers in Model_Wrap.layers:
    layers.trainable = False

Model_flat = Flatten()(Model_Wrap.output)
last_layer = Dense(5, activation='softmax')(Model_flat)

CNN_Model = Model(inputs=Model_Wrap.input, outputs=last_layer, )
CNN_Model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(0.0001))

CNN_Model.summary()

CNN_Model.fit_generator(Train_Data1, epochs=50)

CNN_Model.evaluate(Train_Data1)
CNN_Model.evaluate_generator(Train_Data2)
CNN_Model.evaluate(Test_Data)

CNN_Model.save('CNN_Plant_Recognition.h5')
