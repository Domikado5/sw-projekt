# Facial Attributes Recognizer
## Table of contents
* [Description](#description)
* [Dataset](#dataset)
* [Model](#model)
* [Installation](#installation)
* [Graphs and images](#graphs-and-images)
* [TODO](#todo)

## Description
This project is a part of the Embedded Systems classes. Its main goal is to recognize facial attributes (such as big lips, eyeglasses etc) from a photo. The script is run on a Raspberry Pi which has a USB camera connected to it.

## Dataset
The dataset used for training contains around 202k 218x178 RGB pictures of different celebrities and the facial attributes of the corresponding celebrity (there are 40 different attributes).  
The dataset can be downloaded from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (Align&Cropped Images and Attributes Annotations) **OR** from [kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset).


## Model
1. Description

    - To recognize the attributes we used a deep CNN. It contains 3 convolutional layers, 3 pooling layers and 2 dense layers. To regularize the model we used batch normalization after every pooling layer and dropout layers after every dense layer. All activation functions inside the neural network are leaky ReLUs, the output activation function is sigmoid.

    - As an optimizer we used Adam optimizer, and as the loss - binary crossentropy.  

    - The metrics used to evaluate the performance of the model were AUC, recall and precision.

2. Architecture  
![](/additional/model.png)

## Installation
1. Create virtual environment  
```python -m venv venv```  
Linux or MacOS: ```source venv/bin/activate```  
Windows (in cmd): ```venv/Scripts/activate.bat```  
2. Install libraries  
```pip install -r requirements.txt```  
3. Make sure the model is inside the main directory (where main.py is located) and go to the main directory
4. Run the script  
```python main.py```

## Graphs and images

## TODO
- improve the neural network
- make GUI
- implement face recognition so that the model will recognize known people