### Porosity Permeability Prediction using Transfer Learning
This repository contains code for predicting the porosity and permeability of carbonate core plugs using deep learning techniques and transfer learning.
The code utilizes convolutional neural networks (CNNs) and various pre-trained models to perform the prediction.
### Dataset.
The dataset consists of images of carbonate core plugs, along with their corresponding porosity and permeability values.
The images are stored in a directory specified by the image_path in the code. 
The file paths, porosity, and permeability values are extracted from the image file names and stored in a Pandas data frame for further processing (split and predict).
### Pre-processing
Before training the models, the dataset is split into training and testing sets using the train_test_split function from sci-kit-learn. 
The images are then preprocessed using the ImageDataGenerator class from TensorFlow, which rescales the pixel values between 0 and 1.
### Models
The code includes several models for porosity and permeability prediction:
CNN Model without Transfer Learning: This model is a custom CNN architecture consisting of convolutional, pooling, and dense layers.
VGG-19, VGG-16, InceptionV3, ResNet50, MobileNet, and MobileNet-V2 Pre-Trained Models: 
These models use pre-trained convolutional neural network architectures with weights initialized from the ImageNet dataset. 
The pre-trained models are then augmented with additional layers for fine-tuning the prediction tasks.
### Training and Evaluation
The models are compiled with an optimizer, loss function, and evaluation metrics. 
The fit_model function is used to train the models on the training dataset and evaluate their performance on the validation set. 
The models are trained for a specified number of epochs.
After training, the models are evaluated on the testing dataset using the root mean squared error (RMSE) metrics. 
The predictions are compared to the true values, and the score is calculated to assess the goodness of fit.
### Results
The predictions of each model on the testing dataset are saved to separate CSV files.
These files contain the true porosity and permeability values along with the predicted values for comparison and analysis.
### Execution Time
The execution time for training and prediction is measured AND printed for each step to provide an estimate of the computational requirements of the code.
### Feel free to modify and experiment with the code to suit your specific problem.
