import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import cv2
import time
import copy
import my_dataset
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, model_name, data_path, test_name_to_path, idx_to_class):
    ### Run test through the model
    # img_order_path = '/content/drive/Shareddrives/Stuff/Datasets/2021VRDL_HW1_dataset/testing_img_order.txt'
    img_order_path = data_path + 'testing_img_order.txt'
    ans_path = "models/" + model_name
    # ans_path = '/content/drive/Shareddrives/Stuff/Datasets/2021VRDL_HW1_dataset/answer_resnet50.txt'
    with open(img_order_path) as f:
        test_images = [x.strip() for x in f.readlines()]
    submission = []

    # val_features, val_labels = next(iter(dataloaders['val']))
    # print(f"Feature batch shape: {val_features.size()}")
    # print(f"Labels batch shape: {val_labels.size()}")
    # train_features, train_labels = next(iter(train_dataloader))
    for img in test_images:  # image order is important to your result
        # print(img)
        path = test_name_to_path[img]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = my_dataset.test_transforms(image=image)["image"]
        model.eval()
        inputs = inputs.reshape(1, 3, 300, 300)
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

def save_model(model, optimizer, criterion, exp_lr_scheduler, model_name, epochs):

    model_name = model_name+ '_'+str(epochs) + 'model.pt'
    save_path = 'models/'+model_name
    torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            'scheduler': exp_lr_scheduler,
            }, save_path)
    return model_name

def main():

    dataset_path = "C:\\workspace\\Uni\\Courses\\STCVDL\\HW\\HW1_ImageClassification\\2021VRDL_HW1_datasets\\"
    (dataloaders, dataset_sizes, classes, test_name_to_path, idx_to_class, test_name_to_path) = my_dataset.compose_my_dataset(dataset_path)

    # model_ft = models.resnet34(pretrained=True)
    model_name = 'resnet50'
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(classes))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    epochs = 30
    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.8)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0005)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.5)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=epochs)

    
    model_name = save_model(model_ft, optimizer_ft, criterion, exp_lr_scheduler, model_name, epochs)
 
    test_model(model_ft, model_name, dataset_path, test_name_to_path, idx_to_class)

    return 


if __name__=="__main__":

    main()