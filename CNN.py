# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 21:56:20 2022

@author: krama
"""
import time
import os.path
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

image_path= Path("/home/kunet.ae/100053289/HPCAug2022/imagedir/")
print(image_path)

filepaths = pd.Series(list(image_path.glob(r'**/*.jpg')), name='Filepath').astype(str)
features = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='Data')
features = features.str.split("_", n = 1, expand = True)

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(features)
features = pd.DataFrame(x_scaled)
features.rename(columns = {0:'Porosity', 1:'Permeability'}, inplace = True)
images = pd.concat([filepaths, features], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)

train_df, test_df = train_test_split(images, test_size=0.1, shuffle=True, random_state=1)

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.1)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train=train_gen.flow_from_dataframe(dataframe=train_df, x_col="Filepath", y_col=['Porosity', 'Permeability'],
                                    target_size=(224,224), color_mode="rgb", class_mode="raw", batch_size=32,
                                    shuffle=True, seed=42, subset="training")

test=test_gen.flow_from_dataframe(dataframe=test_df, x_col="Filepath", y_col=['Porosity', 'Permeability'],
                                  target_size=(224,224), color_mode="rgb", class_mode="raw", batch_size=32, shuffle=False)

val_images = train_gen.flow_from_dataframe(dataframe=train_df, x_col='Filepath', y_col=['Porosity', 'Permeability'],
                                           target_size=(224,224), color_mode='rgb', class_mode='raw', batch_size=32,
                                           shuffle=True, seed=42, subset='validation')

type(test.labels)

# model init
def compile_model(model):
    opt = tf.keras.optimizers.Adagrad(learning_rate = 1e-3)
    
    model.compile(
        optimizer=opt, # adam optimalizáció
        loss='mse', # mean squared error hibafüggvény, mivel regresszóként kezeljük
        metrics = ['mean_absolute_error']
    )

class Callback(tf.keras.callbacks.Callback):
    SHOW_NUMBER = 3
    counter = 0
    epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        if self.counter == self.SHOW_NUMBER or self.epoch == 1:
            print('Epoch: ' + str(self.epoch) + ' loss: ' + str(logs['loss']))
            if self.epoch > 1:
                self.counter = 0
        self.counter += 1
        
# modell training
def fit_model(model):
    history = model.fit(train, validation_data=val_images, epochs=200)
#        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True), Callback()])
    return history

# model train
def train_model(model):
    compile_model(model)
    history = fit_model(model)
    # plot_history(history)
    evaluate_model(model)

# # loss-epoch
#def plot_history(history):
#   history_frame = pd.DataFrame(history.history)
#   history_frame.loc[:, ['loss', 'val_loss']].plot()
    
# model RMSE and R^2 score
def evaluate_model(model):
    predicted_ages = np.squeeze(model.predict(test))
    true_ages = test.labels

#    rmse = np.sqrt(model.evaluate(test, verbose=0))
#    print("Test RMSE: {:.5f}".format(rmse))

    ## R-Squared Evaluation
    def compute_r2_numpy(y_true, y_predicted):
        Nr = np.sum(np.square(y_true-y_predicted))
        Dr = np.sum(np.square(y_true-np.mean(y_true)))
        R_squared = 1-(Nr/Dr)
        return R_squared 
    
    def compute_r2(y_true, y_predicted):
        sse = sum((y_true - y_predicted)**2)
        tse = (len(y_true) - 1) * np.var(y_true, ddof=1)
        r2_score = 1 - (sse / tse)
        return r2_score, sse, tse
    
    
    R_square = compute_r2_numpy(true_ages, predicted_ages) 
    print('Coefficient of Determination', R_square)
    
    R22_score = r2_score(true_ages, predicted_ages)
    print('r2_score: {0}'.format(R22_score))  
    
    R2_score,_,_ = compute_r2(true_ages, predicted_ages)
    print('r2_score: {0}'.format(R2_score))
    


'''CNN Model WITHOUT Transfer learning'''    
my_model2 = tf.keras.Sequential([
    layers.InputLayer(input_shape=[224,224,3]),
    
    #Block One
    layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(),
    layers.Dropout(0.2),
    
    #Block Two
    layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(),
    layers.Dropout(0.2),
    
    
    #Block Three
    layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(),
#    layers.Dropout(0.2),
#    
#    #Block Four
#    layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
#    layers.BatchNormalization(),
#    layers.Activation('relu'),
#    layers.MaxPool2D(),
#    layers.Dropout(0.2),
    
    #Head
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='sigmoid')])
my_model2.summary()


'''VGG-19 Pre-Trained Model for transfer learning'''
feature_extractor_VGG19 = tf.keras.applications.VGG19(
    input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling='max')
feature_extractor_VGG19.trainable = False

tl_model_VGG19 = keras.Sequential([
    feature_extractor_VGG19,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='sigmoid')])
tl_model_VGG19.summary()


'''VGG-16 Pre-Trained Model for transfer learning'''
feature_extractor_VGG16 = tf.keras.applications.VGG16(
    input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling='max')
feature_extractor_VGG16.trainable = False

tl_model_VGG16 = keras.Sequential([
    feature_extractor_VGG16,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='sigmoid')])
tl_model_VGG16.summary()


'''InceptionV3 Pre-Trained Model for transfer learning'''
feature_extractor_InceptionV3 = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
feature_extractor_InceptionV3.trainable = False

tl_model_InceptionV3 = keras.Sequential([
    feature_extractor_InceptionV3,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='sigmoid')])
tl_model_InceptionV3.summary()


'''ResNet50 Pre-Trained Model for transfer learning'''
feature_extractor_ResNet50 = tf.keras.applications.resnet50.ResNet50(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
feature_extractor_ResNet50.trainable = False

tl_model_ResNet50 = keras.Sequential([
    feature_extractor_ResNet50,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='sigmoid')])
tl_model_ResNet50.summary()


'''MobileNet Pre-Trained Model for transfer learning'''
feature_extractor_MobileNet = tf.keras.applications.mobilenet.MobileNet(
    input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling='max')
feature_extractor_MobileNet.trainable = False

tl_model_MobileNet = keras.Sequential([
    feature_extractor_MobileNet,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='sigmoid')])
tl_model_MobileNet.summary()


'''MobileNet-V2 Pre-Trained Model for transfer learning'''
feature_extractor_MobileNetV2 = tf.keras.applications.mobilenet_v2.MobileNetV2(
    input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling='max')
feature_extractor_MobileNetV2.trainable = False

tl_model_MobileNetV2 = keras.Sequential([
    feature_extractor_MobileNetV2,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='sigmoid')])
tl_model_MobileNetV2.summary()

print('CNN Model')
st = time.time()
train_model(my_model2)
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

print('VGG-19 Pre-Trained Model for transfer learning')
st = time.time()
train_model(tl_model_VGG19) 
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

print('VGG-16 Pre-Trained Model for transfer learning')
st = time.time()
train_model(tl_model_VGG16)
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

print('InceptionV3 Pre-Trained Model for transfer learning')
st = time.time()
train_model(tl_model_InceptionV3)
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

print('ResNet50 Pre-Trained Model for transfer learning')
st = time.time()
train_model(tl_model_ResNet50)
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

print('MobileNet Pre-Trained Model for transfer learning')
st = time.time()
train_model(tl_model_MobileNet)
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

print('MobileNet-V2 Pre-Trained Model for transfer learning')
st = time.time()
train_model(tl_model_MobileNetV2) 
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')


print('Predictions')
st = time.time()
predictions1 = my_model2.predict(test)
predictions2 = tl_model_VGG19.predict(test)
predictions3 = tl_model_VGG16.predict(test)
predictions4 = tl_model_InceptionV3.predict(test)
predictions5 = tl_model_ResNet50.predict(test)
predictions6 = tl_model_MobileNet.predict(test)
predictions7 = tl_model_MobileNetV2.predict(test)
et = time.time()
elapsed_time = et - st
print('Prediction time:', elapsed_time, 'seconds')

test_data = min_max_scaler.inverse_transform(test.labels)

p1_data = min_max_scaler.inverse_transform(predictions1)
p2_data = min_max_scaler.inverse_transform(predictions2)
p3_data = min_max_scaler.inverse_transform(predictions3)
p4_data = min_max_scaler.inverse_transform(predictions4)
p5_data = min_max_scaler.inverse_transform(predictions5)
p6_data = min_max_scaler.inverse_transform(predictions6)
p7_data = min_max_scaler.inverse_transform(predictions7)

pd.DataFrame(test_data).to_csv('test_data.csv') 
pd.DataFrame(p1_data).to_csv('pLL1_data.csv') 
pd.DataFrame(p2_data).to_csv('pLL2_data.csv')  
pd.DataFrame(p3_data).to_csv('pLL3_data.csv')  
pd.DataFrame(p4_data).to_csv('pLL4_data.csv')  
pd.DataFrame(p5_data).to_csv('pLL5_data.csv')  
pd.DataFrame(p6_data).to_csv('pLL6_data.csv')  
pd.DataFrame(p7_data).to_csv('pLL7_data.csv') 
