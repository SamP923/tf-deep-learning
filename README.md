# tf-deep-learning
This repository contains my work from Udacity's Intro to Deep Learning with TensorFlow course.  

Course Link: https://classroom.udacity.com/courses/ud187  


## Contents

### 1. Celsius to Farenheit Converter 
[Project Notebook](Celsius_to_Farenheit.ipynb)  

**Purpose:**  
Proof of concept project for how machine learning works using a linear regression model (predict a single value from input).

Potential expansions: 
- Ability to test multiple numbers at once, rather than one value 
- Find the relationship between other linear equations
- Find the relationship between more complex equations (add more nodes based on complexity?)



### 2. Clothing Classifier 
[Project Notebook](DL_ClothingClassification.ipynb)  

**Purpose:**  
Classify 10 types of clothing from the Fashion MNIST dataset using a simple neural network. Used with 87.84% accuracy on the test dataset.

Dataset Used: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)


#### Linear Regression versus Classification Problems
![](images/regression_v_classification.PNG)

| | Classification | Regression |
|-|----------------|------------|
|Output| List of numbers that represent probabilities for each class| Single Number|
|Example| Fashion MNIST | Celsisu to Fahrenheit|
|Loss | Sparse categorical crossentropy | Mean squared error|
|Last Layer Activation Function| Softmax | None |


### 3. Clothing Classifier using a Convolutional Neural Network
[Project Notebook](CNN_ClothingClassification.ipynb)  

**Purpose:**  
Build and train a convolutional neural network (CNN) to classify images of clothing. This model is trained on 60,000 images that include 10 types of articles of clothing. This project expands on the previous investigation into classifying clothing using neural networks, except we are now using convolutions for higher performance. Uses two convolution filters and MaxPooling with 91.72% accuracy on the test dataset. This is an improvement from using a single hidden Dense layer.  

Dataset Used: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)  

Project based on [TensorFlow's classification example](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l04c01_image_classification_with_cnns.ipynb)  



### 4. Dogs and Cats Image Classifier
[Project without Image Augmentation](Dogs_vs_Cats_wo_Augmentation.ipynb)  
[Project with Image Augmentation](Dogs_vs_Cats_W_Augmentation.ipynb)  

**Purpose:**  
Classify dogs and cats with the Kaggle dataset using a convolutional neural network and image augmentation.

Dataset Used: [filtered version of Dogs vs. Cats dataset from Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)  

Project based on [TensorFlow's classification example 1](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c01_dogs_vs_cats_without_augmentation.ipynb) and [example 2](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c02_dogs_vs_cats_with_augmentation.ipynb)  

General machine learning workflow
1. Examine and understand data
2. Build an input pipeline
3. Build our model
4. Train our model
5. Test our model
6. Improve our model/Repeat the process


### 5. Flower Classification using CNNs
[Project Notebook](Flower_Classifier_CNNs.ipynb)  

**Purpose:**  
Classify images of flowers with a convolutional neural network using the `tf.keras.Sequential` model and load data using the `ImageDataGenerator` class.  

Dataset Used: [Flower dataset from Google](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)  

Project based on [TensorFlow's classification example](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c03_exercise_flowers_with_data_augmentation.ipynb#scrollTo=OYmOylPlVrVt)



### 6. Dogs and Cats Classifier Revisited
[Project Notebook](Dogs_vs_Cats_TransferLearning.ipynb)  

**Purpose:**  
Use transfer learning to classify images of cats and dogs using [TensorFlow Hub](https://www.tensorflow.org/hub), MobileNet models and the Dogs vs. Cats dataset.  

Concepts Covered:
1. Use a TensorFlow Hub model for prediction
2. Use a TensorFlow Hub model for Dogs vs. Cats dataset
3. Do simple transfer learning with TensorFlow Hub


Dataset Used: [Dogs vs Cats dataset](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)  

Project based on [TensorFlow's transfer learning example](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l06c01_tensorflow_hub_and_transfer_learning.ipynb)  



### 7. Flower Classifier Revisited
[Project Notebook](Flower_Classifier_TransferLearning.ipynb)  

**Purpose:**  
Classify images of flowers with transfer learning using TensorFlow Hub, Google's Flowers Dataset, and MobileNet v2.  

Dataset Used: [Flower dataset from Google](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)  

Project based on [TensorFlow's classification example](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l06c02_exercise_flowers_with_transfer_learning.ipynb)

### 8. Time Series Forecasting
[Project Folder](tfs_examples/)  

**Purpose:**  
Learn implementations of time series forecasting using various architectures. Architectures include:  
1. Linear model
2. Dense models with multiple layers
3. Simple Recurrent Nerual Network (RNN) with simple RNN cells
4. Sequence to vector
5. Sequence to sequence
6. [Stateless RNN](tsf_examples\Forecasting_w_RNNs.ipnyb)
7. [Stateful RNN](tsf_examples\Forecasting_w_Stateful_RNN.ipynb)
8. [Long-Short Term Memory Cells](tfs_examples\Forecasting_w_LSTM.ipynb)
9. [Convolutional Neural Network (CNN)](tsf_example\Forecasting_w_CNNs.ipnyb)



## License

The licenses used by TensorFlow, the makers of the course, have been included in the [LICENSE](license) file in this repository. 
