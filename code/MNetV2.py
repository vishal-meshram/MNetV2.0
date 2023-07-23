import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import tensorflow as tf

from tensorflow.keras import layers,models
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.nasnet import NASNetMobile
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.applications.nasnet import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
from tensorflow.keras import optimizers


#PHASE 1:- DataSet Preparation
# create a array of classes names
filenameAll = os.listdir("/data/AllPhotos") # dataset path.
categoryAll = []
for filename in filenameAll:
  if("Anulom" in filename):
    categoryAll.append("AnulomVilom")
  if("Brahmari" in filename):
    categoryAll.append("Brahmari")
  if("Chandra" in filename):
    categoryAll.append("ChandraBhedan")
  if("Shitali" in filename):
    categoryAll.append("Shitali")
  if("Surya" in filename):
    categoryAll.append("SuryaBhedan")

classifyOnlyPranayamadf = pd.DataFrame({
    'filename': filenameAll,
    'category': categoryAll
})

#classifyOnlyFruitdf = classifyOnlyFruitdf.sort_values(by ='category', ascending = 1)
classifyOnlyPranayamadf = classifyOnlyPranayamadf.sort_values(by ='category', ascending = 1)

#classifyOnlyFruitdf['category'].value_counts().plot.bar()
classifyOnlyPranayamadf['category'].value_counts().plot.bar()

train_df, test_df = train_test_split(classifyOnlyPranayamadf, test_size=0.20, random_state=42) #cosider 80:20 for training and testing
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = test_df.shape[0]
batch_size=32

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

#map the data frames and Images
train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/data/AllPhotos", 
    x_col="filename",
    y_col="category",
    target_size=(224, 224),
    class_mode="categorical",
    batch_size=batch_size
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_df, 
    "/data/AllPhotos", 
    x_col='filename',
    y_col='category',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)
                                      

#PHASE 2: Model architecture
#select pretrained model as MobileNetV2
pretrainedModel = MobileNetV2(input_shape=(224,224,3), weights="imagenet", include_top=False)
for layer in pretrainedModel.layers:
  layer.trainable = False
last_output = pretrainedModel.output
x = layers.GlobalAveragePooling2D()(last_output)
x = layers.Dense(1024, activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x = layers.Dropout(0.25)(x)
x=layers.Dense(1024,activation='relu')(x) #dense layer 2
x = layers.Dropout(0.25)(x)
x=layers.Dense(512,activation='relu')(x) #dense layer 3
x = layers.Dropout(0.25)(x)
preds=layers.Dense(5,activation='softmax')(x) #final layer with softmax activation

model = Model(inputs=pretrainedModel.input, outputs=preds)

#to print our model layes use model isntead of pretrainend 
for i, layer in enumerate(pretrainedModel.layers):
   print(i, layer.name)

# Freez initial layers of the model
for layer in model.layers[:20]:
   layer.trainable = False

for layer in model.layers[20:]:
   layer.trainable = True        
                                                        
#we have used Adam optimizer, short for "Adaptive Moment Estimation optimizer".
base_learning_rate = 0.0001
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()

# solution for error +  None of the MLIR optimization passes are enabled (registered 2)
tf.config.optimizer.set_jit(True)

#Train the Model
history = model.fit(train_generator,
                    validation_data=test_generator,
                    steps_per_epoch=total_train/batch_size,
                    epochs=5,
                    validation_steps=total_validate/batch_size,
                    verbose=1)

#--------------------------------------------------------------------------------------
filenames = os.listdir("/data/AllPhotos")
categoryRightWrong = []
for filename in filenameAll:
  if("Right" in filename):
    categoryRightWrong.append("Right")
  if("Wrong" in filename):
    categoryRightWrong.append("Wrong")
  

classifyRightWrongdf = pd.DataFrame({
    'filename': filenameAll,
    'category': categoryRightWrong
})

classifyRightWrongdf = classifyRightWrongdf.sort_values(by ='category', ascending = 1)

train_df, test_df = train_test_split(classifyRightWrongdf, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = test_df.shape[0]

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

#map the data frames and Images
train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/data/AllPhotos", 
    x_col="filename",
    y_col="category",
    target_size=(224, 224),
    class_mode="categorical",
    batch_size=batch_size
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_df, 
    "/data/AllPhotos", 
    x_col='filename',
    y_col='category',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)   

pretrainedModel = NASNetMobile(input_shape=(224,224,3), weights="imagenet", include_top=False)
for layer in pretrainedModel.layers:
  layer.trainable = False
last_output = pretrainedModel.output
# print(last_output)
x = layers.GlobalAveragePooling2D()(last_output)
x = layers.Dense(1024, activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x = layers.Dropout(0.25)(x)
x=layers.Dense(1024,activation='relu')(x) #dense layer 2
x = layers.Dropout(0.25)(x)
x=layers.Dense(512,activation='relu')(x) #dense layer 3
x = layers.Dropout(0.25)(x)
preds=layers.Dense(2,activation='softmax')(x) #final layer with softmax activation

modelRW = Model(inputs=pretrainedModel.input, outputs=preds) 

#to print our model layes use model instead of pretrainend 
for i, layer in enumerate(pretrainedModel.layers):
   print(i, layer.name)

# Freez the first few layers and retrain the model again 
for layer in modelRW.layers[:20]:
   layer.trainable = False

for layer in modelRW.layers[20:]:
   layer.trainable = True        

#we have used Adam optimizer, short for "Adaptive Moment Estimation optimizer".
base_learning_rate = 0.0001
modelRW.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss='categorical_crossentropy', metrics=['accuracy'])

modelRW.summary()

history = modelRW.fit(train_generator,
                              validation_data=test_generator,
                              steps_per_epoch=total_train/batch_size,
                              epochs=5,
                              validation_steps=total_validate/batch_size,
                              verbose=1)


classes = ['Right', 'Wrong']
PranayamaClasses = ['AnulomVilom','Brahmari', 'ChandraBhedan', 'Shitali', 'SuryaBhedan']

img_path = '/data/TestingImages/Chandra_Bhedan_Pranayam_Right_37.jpg'
img = image.load_img(img_path, target_size=(224, 224, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

PranayamaPreds = model.predict([x])
RightWrongPreds = modelRW.predict([x])

print(PranayamaPreds)
print(RightWrongPreds)

print("Pranayama Prediction Class : " + PranayamaClasses[np.argmax(PranayamaPreds)])
print("Right Wrong Prediction Class : " + classes[np.argmax(RightWrongPreds)])

model.save("results/MNetV2_Pranayama.h5")
modelRW.save("results/MNetV2_modelRW.h5")

