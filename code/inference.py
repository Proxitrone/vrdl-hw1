import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import cv2
import time
import copy
import utils
import numpy as np
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_model(model, model_name, data_path, test_name_to_path, idx_to_class):
    ### Run test through the model
    # img_order_path = '/content/drive/Shareddrives/Stuff/Datasets/2021VRDL_HW1_dataset/testing_img_order.txt'
    # img_order_path = data_path + 'testing_img_order.txt'
    img_order_path = os.path.join(data_path, 'testing_img_order.txt')
    ans_name = model_name+'_answer.txt'
    ans_path = os.path.join("models", ans_name)
    # ans_path = "models/" + model_name
    # ans_path = '/content/drive/Shareddrives/Stuff/Datasets/2021VRDL_HW1_dataset/answer_resnet50.txt'

    with open(img_order_path) as f:
        test_images = [x.strip() for x in f.readlines()]
    submission = []

    for img in test_images:  # image order is important to your result
        # print(img)
        path = test_name_to_path[img]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = utils.test_transforms(image=image)["image"]
        model.eval()
        inputs = inputs.reshape(1, 3, 224, 224)
        # print(inputs.size())
        inputs = inputs.to(device)
        # print(model_ft)
        # model_ft.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        preds = preds.to('cpu').item()
        # print(preds.item())
        label = idx_to_class[preds]
        # print(label)
        submission.append([img, label])
    # print(submission)
    np.savetxt(ans_path, submission, fmt='%s')
    return


if __name__=="__main__":
    # dataset_path = "C:\\workspace\\Uni\\Courses\\STCVDL\\HW\\HW1_ImageClassification\\2021VRDL_HW1_datasets\\"
    (dataloaders, dataset_sizes, classes, test_name_to_path, idx_to_class) = utils.compose_my_dataset(utils.dataset_path)

    model_name = 'resnet152_62model.pt'
    # model_name = 'VGG16_50model.pt'
    (model_ft, optimizer_ft, epoch_num, criterion, exp_lr_scheduler) = utils.load_model(model_name, classes)

    test_model(model_ft, model_name, utils.dataset_path, test_name_to_path, idx_to_class)