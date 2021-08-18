################################################################################
#
# DESCRIPTION
#
#    Image classification in PyTorch for ImageNet reduced to 100 classes and
#    down sampled such that the short side is 64 pixels and the long side is
#    >= 64 pixels
#
#    This script achieved a best accuracy of 70.25% on epoch 60 with a learning
#    rate at that point of 0.000010 and time required for each epoch of ~ 194 s
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Run all
#
# NOTES
#
#    0. For a mapping of category names to directory names see:
#       https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
#
#    1. The original 2012 ImageNet images are down sampled such that their short
#       side is 64 pixels (the other side is >= 64 pixels) and only 100 of the
#       original 1000 classes are kept.
#
#    2. Build and train a RegNetX image classifier modified as follows:
#
#       - Set stride = 1 (instead of stride = 2) in the stem
#       - Replace the first stride = 2 down sampling building block in the
#         original network by a stride = 1 normal building block
#       - The fully connected layer in the decoder outputs 100 classes instead
#         of 1000 classes
#
#       The original RegNetX takes in 3x224x224 input images and generates Nx7x7
#       feature maps before the decoder, this modified RegNetX will take in
#       3x56x56 input images and generate Nx7x7 feature maps before the decoder.
#       For reference, an implementation of this network took ~ 112 s per epoch
#       for training, validation and checkpoint saving on Sep 27, 2020 using a
#       free GPU runtime in Google Colab.
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

# torch
import torch
import torch.nn       as     nn
import torch.optim    as     optim
from   torch.autograd import Function
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


# torch utils
import torchvision
import torchvision.transforms as transforms


# additional libraries
import os
import urllib.request
import zipfile
import time
import math
import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd
import datetime

################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_DIR_1        = 'data'
DATA_DIR_2        = 'data/imagenet64'
DATA_DIR_TRAIN    = 'data/imagenet64/train'
DATA_DIR_TEST     = 'data/imagenet64/val'
DATA_FILE_TRAIN_1 = 'Train1.zip'
DATA_FILE_TRAIN_2 = 'Train2.zip'
DATA_FILE_TRAIN_3 = 'Train3.zip'
DATA_FILE_TRAIN_4 = 'Train4.zip'
DATA_FILE_TRAIN_5 = 'Train5.zip'
DATA_FILE_TEST_1  = 'Val1.zip'
DATA_URL_TRAIN_1  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train1.zip'
DATA_URL_TRAIN_2  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train2.zip'
DATA_URL_TRAIN_3  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train3.zip'
DATA_URL_TRAIN_4  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train4.zip'
DATA_URL_TRAIN_5  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train5.zip'
DATA_URL_TEST_1   = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Val1.zip'
DATA_BATCH_SIZE   = 256
DATA_NUM_WORKERS  = 4
DATA_NUM_CHANNELS = 3
DATA_NUM_CLASSES  = 100
DATA_RESIZE       = 64
DATA_CROP         = 56

DATA_MEAN         = (0.485, 0.456, 0.406)
DATA_STD_DEV      = (0.229, 0.224, 0.225)

# model
depth_list = [1, 1, 4, 7]
width_list = [24, 56, 152, 368]
stride = 1
group_width = 8
bottleneck_ratio = 1
initial_width = 24

# training (linear warm up with cosine decay learning rate)
TRAINING_LR_MAX          = 0.001
TRAINING_LR_INIT_SCALE   = 0.01
TRAINING_LR_INIT_EPOCHS  = 5
TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_FINAL_EPOCHS = 55

TRAINING_NUM_EPOCHS      = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
TRAINING_LR_INIT         = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL        = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE

# file
FILE_NAME = 'proj2_HXV190008.pt'
FILE_SAVE = 0
FILE_LOAD = 0

################################################################################
#
# DATA
#
################################################################################

# create a local directory structure for data storage
if (os.path.exists(DATA_DIR_1) == False):
    os.mkdir(DATA_DIR_1)
if (os.path.exists(DATA_DIR_2) == False):
    os.mkdir(DATA_DIR_2)
if (os.path.exists(DATA_DIR_TRAIN) == False):
    os.mkdir(DATA_DIR_TRAIN)
if (os.path.exists(DATA_DIR_TEST) == False):
    os.mkdir(DATA_DIR_TEST)

# download data
if (os.path.exists(DATA_FILE_TRAIN_1) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_1, DATA_FILE_TRAIN_1)
if (os.path.exists(DATA_FILE_TRAIN_2) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_2, DATA_FILE_TRAIN_2)
if (os.path.exists(DATA_FILE_TRAIN_3) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_3, DATA_FILE_TRAIN_3)
if (os.path.exists(DATA_FILE_TRAIN_4) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_4, DATA_FILE_TRAIN_4)
if (os.path.exists(DATA_FILE_TRAIN_5) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_5, DATA_FILE_TRAIN_5)
if (os.path.exists(DATA_FILE_TEST_1) == False):
    urllib.request.urlretrieve(DATA_URL_TEST_1, DATA_FILE_TEST_1)

# extract data
with zipfile.ZipFile(DATA_FILE_TRAIN_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_2, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_3, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_4, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_5, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TEST_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TEST)

# transforms
transform_train = transforms.Compose([transforms.RandomResizedCrop(DATA_CROP), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])
#transform_test = transforms.Compose([transforms.RandomResizedCrop(DATA_CROP), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])
transform_test  = transforms.Compose([transforms.Resize(DATA_RESIZE), transforms.CenterCrop(DATA_CROP), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])

# data sets
dataset_train = torchvision.datasets.ImageFolder(DATA_DIR_TRAIN, transform=transform_train)
dataset_test  = torchvision.datasets.ImageFolder(DATA_DIR_TEST,  transform=transform_test)

# data loader
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=DATA_BATCH_SIZE, shuffle=True,  num_workers=DATA_NUM_WORKERS, pin_memory=True, drop_last=True)
dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=DATA_BATCH_SIZE, shuffle=False, num_workers=DATA_NUM_WORKERS, pin_memory=True, drop_last=True)

################################################################################
#
# NETWORK BUILDING BLOCK
#
################################################################################

#Tail/ Stem design
#Modified to have stride = 1
class Tail(nn.Module):

    def __init__(self, DATA_NUM_CHANNELS, initial_width):
        super(Tail, self).__init__()
        self.conv = nn.Conv2d(DATA_NUM_CHANNELS, initial_width, (3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn = nn.BatchNorm2d(initial_width)
        self.rl = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.rl(x)
        return x

#Head design - Average pooling + Fully connected Layer
class Head(nn.Module):

    def __init__(self, DATA_NUM_CHANNELS, DATA_NUM_CLASSES):
        super(Head, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fclayer = nn.Linear(DATA_NUM_CHANNELS, DATA_NUM_CLASSES)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fclayer(x)
        return x

#Building Block
class XBlock(nn.Module):

    def __init__(self, C_in , C_out, bottleneck_ratio, group_width, stride):
        super(XBlock, self).__init__()
        C_local = math.floor(C_out / bottleneck_ratio)
        groups = math.floor(C_local / group_width)

        self.block1_conv = nn.Conv2d(C_in, C_local, (1, 1), stride=1, padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.block1_bn = nn.BatchNorm2d(C_local)

        self.block2_conv = nn.Conv2d(C_local, C_local, (3, 3), stride=stride, groups=groups, padding=(1, 1), bias=False, padding_mode='zeros')
        self.block2_bn = nn.BatchNorm2d(C_local)

        self.block3_conv = nn.Conv2d(C_local, C_out, (1, 1), stride=1, padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.block3_bn = nn.BatchNorm2d(C_out)

        #Identity Block for s = 2
        self.identity = nn.Sequential()
        if stride != 1:
            self.iden_conv = True
            self.identity = nn.Sequential(
                nn.Conv2d(C_in, C_out, (1, 1), stride=2, bias=False),
                nn.BatchNorm2d(C_out)
            )
        else:
          self.identity = None
          self.iden_conv = False

    def forward(self, x):
        x1 = F.relu(self.block1_bn(self.block1_conv(x)))
        x1 = F.relu(self.block2_bn(self.block2_conv(x1)))
        x1 = self.block3_bn(self.block3_conv(x1))
        if (self.iden_conv == True):
            res = self.identity(x)
        else:
            res = x
        # print(res.shape, x1.shape)
        x1 = res + x1
        x1 = F.relu(x1)
        return x1
      
#Stage Design
class Stage(nn.Module):
    def __init__(self, stage_num, initial_width, depth_list, width_list, bottleneck_ratio, group_width, stride):
        super(Stage, self).__init__()
        self.stage_num = stage_num
        self.blocks = nn.Sequential()
        
        #Block Design for Normal Building Block
        if (stage_num == 0):
          self.blocks.add_module("Block{0}", XBlock(initial_width, width_list[stage_num], bottleneck_ratio, group_width, stride))
        
        #Block Design for Down Sampling Building Block
        else:
          self.blocks.add_module("Block{0}", XBlock(initial_width, width_list[stage_num], bottleneck_ratio, group_width, stride = 2))
       
        for i in range(1, depth_list[stage_num]):
          self.blocks.add_module("Block{}".format(i), XBlock(width_list[stage_num], width_list[stage_num], bottleneck_ratio, group_width, stride))

    def forward(self, x):
        x = self.blocks(x)
        return x

################################################################################
#
# NETWORK
#
################################################################################

class Model(nn.Module):
    def __init__(self, initial_width, depth_list, width_list, bottleneck_ratio, group_width, stride):
        super(Model, self).__init__()
        
        #Tail/ Stem
        self.net = nn.Sequential()
        self.net.add_module("Tail", Tail(DATA_NUM_CHANNELS, initial_width))
    
        #Stage 1- 4
        for i in range(0,4):
            self.net.add_module("Stage{}".format(i), Stage(i, initial_width, depth_list, width_list, bottleneck_ratio, group_width, stride))
            initial_width = width_list[i]
        
        #Head
        self.net.add_module("Head", Head(width_list[-1], DATA_NUM_CLASSES))

    def forward(self, x):
        x = self.net(x)
        return x

#Run Model
model = Model(initial_width, depth_list, width_list, bottleneck_ratio, group_width, stride)
print(model)

conv_params = 0
for key in model.modules():
    if (isinstance(key, nn.Conv2d) | isinstance(key, nn.Linear)):
        conv_params += sum(p.numel() for p in key.parameters() if p.requires_grad)
print("Total Convolution Parameters: ", conv_params)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total Trainable Parameters: ", pytorch_total_params)

################################################################################
#
# TRAIN & TEST
#
################################################################################

# start epoch
start_epoch = 0

# learning rate schedule
def lr_schedule(epoch):

    # linear warmup followed by cosine decay
    if epoch < TRAINING_LR_INIT_EPOCHS:
        lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
    else:
        lr = (TRAINING_LR_MAX - TRAINING_LR_FINAL)*max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/(TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + TRAINING_LR_FINAL

    return lr

# error (softmax cross entropy)
criterion = nn.CrossEntropyLoss()

# optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# specify the device as the GPU if present with fallback to the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ",device)

# transfer the network to the device
model.to(device)

# model loading
if FILE_LOAD == 1:
    checkpoint = torch.load(FILE_NAME)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

#Graph Variable
results = pd.DataFrame(columns=['Epochs','Training Data Loss','Testing Data Accuracy'])
# cycle through the epochs
for epoch in range(start_epoch, TRAINING_NUM_EPOCHS):
    print("epoch started:", epoch)
    print(datetime.datetime.now().time())

    # initialize train set statistics
    model.train()
    training_loss = 0.0
    num_batches   = 0

    # set the learning rate for the epoch
    for g in optimizer.param_groups:
        g['lr'] = lr_schedule(epoch)

    # cycle through the train set
    for data in dataloader_train:

        # extract a batch of data and move it to the appropriate device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass, loss, backward pass and weight update
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # update statistics
        training_loss = training_loss + loss.item()
        num_batches   = num_batches + 1

    # initialize test set statistics
    model.eval()
    test_correct = 0
    test_total   = 0

    # no weight update / no gradient needed
    with torch.no_grad():

        # cycle through the test set
        for data in dataloader_test:

            # extract a batch of data and move it to the appropriate device
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass and prediction
            outputs      = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # update test set statistics
            test_total   = test_total + labels.size(0)
            test_correct = test_correct + (predicted == labels).sum().item()

    # epoch statistics
    print('Epoch {0:2d} lr = {1:8.6f} avg loss = {2:8.6f} accuracy = {3:5.2f}'.format(epoch, lr_schedule(epoch), (training_loss/num_batches)/DATA_BATCH_SIZE, (100.0*test_correct/test_total)))
    results.loc[epoch, ['Epochs']] = epoch
    results.loc[epoch, ['Training Data Loss']] = (training_loss/num_batches)/DATA_BATCH_SIZE
    results.loc[epoch, ['Testing Data Accuracy']] = (100.0 * test_correct/test_total)

# model saving
# to use this for checkpointing put this code block inside the training loop at the end (e.g., just indent it 4 spaces)
# and set 'epoch' to the current epoch instead of TRAINING_NUM_EPOCHS - 1; then if there's a crash it will be possible
# to load this checkpoint and restart training from the last complete epoch instead of having to start training at the
# beginning
if FILE_SAVE == 1:
    torch.save({
        'epoch': TRAINING_NUM_EPOCHS - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, FILE_NAME)

results

#Plots for training loss and accuracy wrt to epochs
plt.rcParams["figure.figsize"] = (10,5)
results.plot(x = 'Epochs', y = 'Training Data Loss', color='red',   linewidth=1.0)
plt.xlabel("Epochs")
plt.ylabel("Training Data Loss")
plt.title("Epoch vs Training Data Loss")

results.plot(x = 'Epochs', y = 'Testing Data Accuracy', color='red',   linewidth=1.0)
plt.xlabel("Epochs")
plt.ylabel("Testing Data Accuracy")
plt.title("Epoch vs Testing Data Accuracy")

plt.show()
