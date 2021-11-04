from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import torch.optim as optim
import cv2
import glob
import random
import os



dataset_path = "G:\\Documents\\Uni\\Courses\\STCVDL\\HW1\\2021VRDL_HW1_dataset\\"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#######################################################
#               Define Transforms
#######################################################

#To define an augmentation pipeline, you need to create an instance of the Compose class.
#As an argument to the Compose class, you need to pass a list of augmentations you want to apply. 
#A call to Compose will return a transform function that will perform image augmentation.
#(https://albumentations.ai/docs/getting_started/image_augmentation/)

train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=256),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        A.RandomCrop(height=224, width=224),
        # A.RandomSizedCrop(height=256, width=256)
        # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        # A.RandomBrightnessContrast(p=0.5),
        A.augmentations.geometric.transforms.Perspective(interpolation=3, p=0.5),
        # A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        # A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


####################################################
#       Create Train, Valid and Test sets
####################################################
def init_dataset_info(dataset_path):

    train_data_path = os.path.join(dataset_path, 'train')
    test_data_path = os.path.join(dataset_path, 'test')
    training_labels_path =os.path.join(dataset_path, 'training_labels.txt')
    classes_path = os.path.join(dataset_path, 'classes.txt')
    # train_data_path = '/content/drive/Shareddrives/Stuff/Datasets/2021VRDL_HW1_dataset/train' 
    # test_data_path = '/content/drive/Shareddrives/Stuff/Datasets/2021VRDL_HW1_dataset/test'
    # classes_path = '/content/drive/Shareddrives/Stuff/Datasets/2021VRDL_HW1_dataset/classes.txt'
    # training_labels_path = '/content/drive/Shareddrives/Stuff/Datasets/2021VRDL_HW1_dataset/training_labels.txt'
    train_image_paths = [] #to store image paths in list
    classes = [] #to store class values
    training_labels = []
    #1.
    # get all the paths from train_data_path and append image paths and class to to respective lists

    # print(glob.glob(train_data_path + '/*' ))
    # for data_path in glob.glob(train_data_path ):
    #     # classes.append(data_path.split('/')[-1]) 
    #     train_image_paths.append(glob.glob(data_path + '/*'))
    train_image_paths = glob.glob(train_data_path + '/*' )
    # print(train_image_paths)
    with open(classes_path) as f:
        classes = [x.strip() for x in f.readlines()]
    # classes = [x.split('.') for x in f.readlines()]

    with open(training_labels_path) as f:
        training_labels = [x.strip().split(' ') for x in f.readlines()]

    # print(training_labels)
    # train_image_paths = list(flatten(train_image_paths))
    random.shuffle(train_image_paths)

    print('train_image_path example: ', train_image_paths[0])
    print('class example: ', classes[0])

    #2.
    # copy valid as train paths
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
        # name = i.split('/')[-1]
        (dir, name)=  os.path.split(i)
        test_name_to_path[name] = i
    # print(test_name_to_path) 

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
        # sample_name = image_filepath.split('/')[-1]
        (dir, sample_name) = os.path.split(image_filepath)
        # print(image_filepath)
        # print(sample_name)
        class_name = self.img_to_class[sample_name]
        label = self.class_to_idx[class_name]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
            # image = self.transform(image)
        
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

    image_datasets = {}
    image_datasets['train']=train_dataset
    image_datasets['val']=valid_dataset
    dataloaders = {x:DataLoader(image_datasets[x], batch_size = 8, shuffle = True)
                    for x in ['train', 'val']}

    dataset_sizes = {x:len(image_datasets[x]) for x in ['train', 'val']}

    return (dataloaders, dataset_sizes)


#######################################################
#                  Compose Dataset
#######################################################
def compose_my_dataset(dataset_path):

    (train_image_paths, valid_image_paths, test_image_paths, training_labels, classes) = init_dataset_info(dataset_path)

    (idx_to_class, class_to_idx, img_to_class, test_name_to_path) = create_index_dicts(training_labels, classes, test_image_paths)

    (train_dataset, valid_dataset) = create_datasets(train_image_paths, valid_image_paths, img_to_class, class_to_idx)

    (dataloaders, dataset_sizes) = create_dataloaders(train_dataset, valid_dataset)

    return (dataloaders, dataset_sizes, classes, test_name_to_path, idx_to_class)


#######################################################
#            Define SpinalNet Classifier
#######################################################
class SpinalNet_ResNet(nn.Module):   
    def __init__(self, half_in_size, layer_width, num_class):
        super(SpinalNet_ResNet, self).__init__()
        self.dropout = nn.Dropout(p=0.35)      
        self.fc_spinal_layer1 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width*1, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width*2, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width*3, layer_width),
            #nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            #nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width*4, num_class),)
        
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        xOrgD = self.dropout(x)
        x1 = self.fc_spinal_layer1(x)
        xOrgD = torch.cat([xOrgD, x1], dim=1)
        x2 = self.fc_spinal_layer2(xOrgD)
        xOrgD = torch.cat([xOrgD, x2], dim=1)
        x3 = self.fc_spinal_layer3(xOrgD)
        xOrgD = torch.cat([xOrgD, x3], dim=1)
        x4 = self.fc_spinal_layer4(xOrgD)
        x = torch.cat([xOrgD, x4], dim=1)
        x = self.fc_out(x)
        return x 


def save_model(model, optimizer, criterion, exp_lr_scheduler, model_name, epochs):

    model_name = model_name+ '_'+str(epochs) + 'model'
    # save_path = 'models/'+model_name
    save_path = os.path.join('models',model_name+'.pt')
    torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            'scheduler': exp_lr_scheduler,
            }, save_path)
    return model_name


def load_model(model_name, classes):
    ## Load model and optimizer state to resume training
    # model_name = 'resnet34_'+str(epochs) + 'model.pt'
    # save_path = '/content/drive/Shareddrives/Stuff/Datasets/2021VRDL_HW1_dataset/'+model_name
    save_path = os.path.join('models',model_name)
    # device = torch.device("cuda")

    ## Resnet152
    model =  models.resnet152()
    num_ftrs = model.fc.in_features

    half_in_size = num_ftrs
    layer_width = 300 #Small for Resnet, large for VGG
    num_class=len(classes)

    model.fc = SpinalNet_ResNet(half_in_size, layer_width, num_class)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    checkpoint = torch.load(save_path)
    print(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_num = checkpoint['epoch']
    print("Prev epoch:", epoch_num)
    criterion = checkpoint['loss']
    lr_scheduler = checkpoint['scheduler']
    model = model.to(device)
    return (model, optimizer, epoch_num, criterion, lr_scheduler)

