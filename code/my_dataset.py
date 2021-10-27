
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import glob
import numpy
import random

# import matplotlib.pyplot as plt

#######################################################
#               Define Transforms
#######################################################

#To define an augmentation pipeline, you need to create an instance of the Compose class.
#As an argument to the Compose class, you need to pass a list of augmentations you want to apply. 
#A call to Compose will return a transform function that will perform image augmentation.
#(https://albumentations.ai/docs/getting_started/image_augmentation/)

train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        A.RandomCrop(height=300, width=300),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.CenterCrop(height=300, width=300),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


####################################################
#       Create Train, Valid and Test sets
####################################################
def init_dataset_info(dataset_path):

    train_data_path = dataset_path + 'train'
    test_data_path = dataset_path + 'test'
    training_labels_path = dataset_path + 'training_labels.txt'
    classes_path = dataset_path + 'classes.txt'
    # train_data_path = '/content/drive/Shareddrives/Stuff/Datasets/2021VRDL_HW1_dataset/train' 
    # test_data_path = '/content/drive/Shareddrives/Stuff/Datasets/2021VRDL_HW1_dataset/test'
    # classes_path = '/content/drive/Shareddrives/Stuff/Datasets/2021VRDL_HW1_dataset/classes.txt'
    # training_labels_path = '/content/drive/Shareddrives/Stuff/Datasets/2021VRDL_HW1_dataset/training_labels.txt'
    train_image_paths = [] #to store image paths in list
    classes = [] #to store class values
    training_labels = []
    #1.
    # get all the paths from train_data_path and append image paths and class to to respective lists
    # eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
    # eg. class -> 26.Pont_du_Gard
    # print(glob.glob(train_data_path + '/*' ))
    # for data_path in glob.glob(train_data_path ):
    #     # classes.append(data_path.split('/')[-1]) 
    #     train_image_paths.append(glob.glob(data_path + '/*'))
    train_image_paths = glob.glob(train_data_path + '/*' )

    with open(classes_path) as f:
        classes = [x.strip() for x in f.readlines()]
    # classes = [x.split('.') for x in f.readlines()]

    with open(training_labels_path) as f:
        training_labels = [x.strip().split(' ') for x in f.readlines()]

    print(training_labels)
    # train_image_paths = list(flatten(train_image_paths))
    random.shuffle(train_image_paths)

    print('train_image_path example: ', train_image_paths[0])
    print('class example: ', classes[0])

    #2.
    # split train valid from train paths (80,20)
    # train_image_paths, valid_image_paths = train_image_paths[:int(0.9*len(train_image_paths))], train_image_paths[int(0.9*len(train_image_paths)):] 
    train_image_paths, valid_image_paths = train_image_paths, train_image_paths 

    #3.
    # create the test_image_paths
    test_image_paths = []
    test_image_paths = glob.glob(test_data_path + '/*' )
    # for data_path in glob.glob(test_data_path ):
    #     test_image_paths.append(glob.glob(data_path + '/*'))

    # test_image_paths = list(flatten(test_image_paths))

    print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))

    return (train_image_paths, valid_image_paths, test_image_paths, training_labels, classes)

#######################################################
#      Create dictionary for class indexes
#######################################################
def create_index_dicts(training_labels, classes, test_image_paths):
    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}
    img_to_class = {}
    for i in range(len(training_labels)):
        for j in range(len(training_labels[i])):
            if j == 0:
                key = training_labels[i][j]
            else:
                value = training_labels[i][j]
        img_to_class[key]=value

    ## Create name to path dictionary
    test_name_to_path={}
    # test_image_paths
    for i in test_image_paths:
        # path = test_image_paths[i]
        name = i.split('/')[-1]
        test_name_to_path[name] = i
    # print(img_to_class) 
    return (idx_to_class, class_to_idx, img_to_class, test_name_to_path)


#######################################################
#               Define Dataset Class
#######################################################

class BirdSpeciesDataset(Dataset):
    def __init__(self, image_paths, img_to_class, class_to_idx, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        self.img_to_class=img_to_class
        self.class_to_idx=class_to_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # label = image_filepath.split('/')[-2]
        sample_name = image_filepath.split('/')[-1]
        class_name = self.img_to_class[sample_name]
        label = self.class_to_idx[class_name]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label
    
#######################################################
#                  Create Dataset
#######################################################

def create_datasets(train_image_paths, valid_image_paths, img_to_class, class_to_idx):
    train_dataset = BirdSpeciesDataset(train_image_paths, img_to_class, class_to_idx,train_transforms)
    valid_dataset = BirdSpeciesDataset(valid_image_paths, img_to_class, class_to_idx,test_transforms) #test transforms are applied
    # test_dataset = BirdSpeciesDataset(test_image_paths,test_transforms)
    return (train_dataset, valid_dataset)

#######################################################
#                  Define Dataloaders
#######################################################
def create_dataloaders(train_dataset, valid_dataset):
    # train_loader = DataLoader(
    #     train_dataset, batch_size=32, shuffle=True
    # )

    # valid_loader = DataLoader(
    #     valid_dataset, batch_size=32, shuffle=True
    # )


    # test_loader = DataLoader(
    #     test_dataset, batch_size=32, shuffle=False
    # )
    image_datasets = {}
    image_datasets['train']=train_dataset
    image_datasets['val']=valid_dataset
    dataloaders = {x:DataLoader(image_datasets[x], batch_size = 32, shuffle = True)
                    for x in ['train', 'val']}

    dataset_sizes = {x:len(image_datasets[x]) for x in ['train', 'val']}

    return (dataloaders, dataset_sizes)

def compose_my_dataset(dataset_path):

    (train_image_paths, valid_image_paths, test_image_paths, training_labels, classes) = init_dataset_info(dataset_path)

    (idx_to_class, class_to_idx, img_to_class, test_name_to_path) = create_index_dicts(training_labels, classes, test_image_paths)

    (train_dataset, valid_dataset) = create_datasets(train_image_paths, valid_image_paths, img_to_class, class_to_idx)

    (dataloaders, dataset_sizes) = create_dataloaders(train_dataset, valid_dataset)

    return (dataloaders, dataset_sizes, classes, test_name_to_path)


# def save_model(model, optimizer, criterion, exp_lr_scheduler, model_name, save_path, epochs,):

#     model_name = 'resnet50_'+str(epochs) + 'model.pt'
#     save_path = '/content/drive/Shareddrives/Stuff/Datasets/2021VRDL_HW1_dataset/'+model_name
#     torch.save({
#             'epoch': epochs,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': criterion,
#             'scheduler': exp_lr_scheduler,
#             }, save_path)
