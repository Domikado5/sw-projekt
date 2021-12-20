# Facial Attributes Recognizer
## Table of contents
* [Description](#description)
* [Dataset](#dataset)
* [Model](#model)
    * [Architecture](#architecture)
* [Installation](#installation)
* [Graphs and images](#graphs-and-images)
    * [Accuracy](#accuracy)
    * [Classification report](#classification-report)
* [Encountered difficulties](#encountered-difficulties)
* [Future development](#future-development)
* [Tech stack](#tech-stack)
* [TODO](#todo)

## Description
This project is a part of the Embedded Systems classes. Its main goal is to recognize facial attributes (such as big lips, eyeglasses etc) from a photo. The script is run on a Raspberry Pi which has a USB camera connected to it.

## Dataset
The dataset used for training contains around 202k 218x178 RGB pictures of different celebrities and the facial attributes of the corresponding celebrity (there are 40 different attributes).  
182k pictures were used for training, the remaining ~20k were used for validation.  
The dataset can be downloaded from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (Align&Cropped Images and Attributes Annotations) **OR** from [kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset).


## Model
### Architecture  
The model consists of a previously trained model (MobileNetV2) and two, top-most Dense layers. To prevent overfitting, batch normalization, dropout and data augmentation were used.  
The loss value after evaluation on the test set was **0.21** and the binary accuracy was **0.91** (though the accuracy can be misleading sometimes because some labels were very imbalanced).  

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

### Accuracy
![](/additional/Accuracy_plot.png)

### Classification report
![](/additional/classification_report.png)

## Encountered difficulties
- finding the "sweet spot" between a too weak and a too complex model (after all ~5 models were trained)  

## Future development
- improving a neural network may lead to a better performance  

## Tech stack
- Python:
    - Tenforflow
    - OpenCV
    - NumPy

## TODO
- [x] improve the neural network
- [x] make GUI
- [ ] implement face recognition so that the model will recognize known people