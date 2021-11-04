import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import copy
import utils
import numpy as np
import inference

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # for fold, (train_idx, val_idx) in enumerate(splits.splot(np.arrange(len(train_dataset)))) 
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over K-folds of sampled data
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # Remove scheduler
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())

        print()
    best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Final val Acc: {:4f}'.format(epoch_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    # dataset_path = "G:\\Documents\\Uni\\Courses\\STCVDL\\HW1\\2021VRDL_HW1_dataset\\"
    # dataset_path = "C:\\workspace\\Uni\\Courses\\STCVDL\\HW\\HW1_ImageClassification\\2021VRDL_HW1_datasets\\"
    (dataloaders, dataset_sizes, classes, test_name_to_path, idx_to_class) = utils.compose_my_dataset(utils.dataset_path)

    ## Resnet152
    model_name = 'resnet152'
    model_ft = models.resnet152(pretrained=True)

    # for param in model_ft.parameters():
    #     param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    half_in_size = num_ftrs
    layer_width = 300 #Small for Resnet, large for VGG
    num_class=len(classes)
    # model_ft.fc = nn.Linear(num_ftrs, len(classes))

    model_ft.fc = utils.SpinalNet_ResNet(half_in_size, layer_width, num_class) 

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    epochs = 62
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.003)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
    print(model_ft)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=epochs)

    model_name = utils.save_model(model_ft, optimizer_ft, criterion, exp_lr_scheduler, model_name, epochs)
 
    inference.test_model(model_ft, model_name, utils.dataset_path, test_name_to_path, idx_to_class)

    return 

def continue_training():

    # dataset_path = "G:\\Documents\\Uni\\Courses\\STCVDL\\HW1\\2021VRDL_HW1_dataset\\"
    # dataset_path = "C:\\workspace\\Uni\\Courses\\STCVDL\\HW\\HW1_ImageClassification\\2021VRDL_HW1_datasets\\"
    (dataloaders, dataset_sizes, classes, test_name_to_path, idx_to_class) = utils.compose_my_dataset(utils.dataset_path)

    model_name = 'resnet152_62model.pt'
    # model_name = 'VGG16_50model.pt'
    (model_ft, optimizer_ft, epoch_num, criterion, exp_lr_scheduler) = utils.load_model(model_name, classes)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)
    epochs = epoch_num + epoch_num
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=epoch_num)

    
    model_name = utils.save_model(model_ft, optimizer_ft, criterion, exp_lr_scheduler, model_name, epochs)
 
    inference.test_model(model_ft, model_name, utils.dataset_path, test_name_to_path, idx_to_class)

    return

if __name__=="__main__":
    
    main()
    # continue_training()
    # dataset_path = "G:\\Documents\\Uni\\Courses\\STCVDL\\HW1\\2021VRDL_HW1_dataset\\"
    # (dataloaders, dataset_sizes, classes, test_name_to_path, idx_to_class) = my_dataset.compose_my_dataset(dataset_path)

    # model_name = 'resnet152_40model.pt'
    # # model_name = 'VGG16_50model.pt'
    # (model_ft, optimizer_ft, epoch_num, criterion, exp_lr_scheduler) = load_model(model_name, classes)

    # test_model(model_ft, model_name, dataset_path, test_name_to_path, idx_to_class)