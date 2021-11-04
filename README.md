# VRDL HW1: Bird Species Image Classification

This repository contains the training and inferencing code for the bird image classification homework challenge.

## Requirements
Python3, Pytorch, Albumentations, OpenCV, Torchvision

## Setup
Create a python virtual environment and run 
```
pip install -r requirements.txt
```
## Dataset preparation: 

Uzip the dataset and internal folders, write the resulting path into utils.py as the `data_path` variable

## Model path
Models will be saved in the models is in `models\` directory.

## Training

To train the model, run 
```
python code\train.py
```
This will result in ``models\resnet152_45model.pt` and `models\resnet152_45model_answer.txt` files
## Reproduce Submission Results

To repeat the inferencing results without training, run
``` 
python code\inference.py
```
The best performing model name is already written there. It will produce a file named `resnet152_45model_answer.txt` in the `models\` folder, which is the inference result. You need to rename it to `answer.txt` and submit it as a .zip archive
