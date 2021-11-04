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
The best performing model is in `models\resnet152_45model.pt`

## Training

To train the model, run 
```
python train.py
```

## Reproduce Submission Results

To repeat the inferencing results, run
``` 
python inference.py
```
The best performing model name is already written there. It will produce a file named `resnet152_45model_answer.txt`, which is the inference result. You need to rename it to `answer.txt` and submit it as a .zip archive
