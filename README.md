# tf-deep-learning
This repository contains my work from Udacity's Intro to Deep Learning with TensorFlow course.  

Course Link: https://classroom.udacity.com/courses/ud187  


## Contents

### 1. Celsius to Farenheit Converter 
Proof of concept project for how machine learning work using a linear regression model (predict a single value from input).

Potential expansions: 
- Ability to test multiple numbers at once, rather than one value 
- Find the relationship between other linear equations
- Find the relationship between more complex equations (add more nodes based on complexity?)


### 2. Clothing Classifier  
Classify 10 types of clothing using the Fashion MNIST dataset.

Potential expansions:
- Work on RPi for live detection?


#### Linear Regression versus Classification Problems

![](images/regression_v_classification.PNG)

| | Classification | Regression |
|-|----------------|------------|
|Output| List of numbers that represent probabilities for each class| Single Number|
|Example| Fashion MNIST | Celsisu to Fahrenheit|
|Loss | Sparse categorical crossentropy | Mean squared error|
|Last Layer Activation Function| Softmax | None |

## License

The licenses used by TensorFlow, the makers of the course, have been included in the [LICENSE](license) file in this repository. 
