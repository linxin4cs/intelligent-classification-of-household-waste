"""### 加载依赖"""
from __future__ import print_function, division
import os
import time
import glob
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


# 标签平滑
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def lin_comb(a, b, epsilon):
    return epsilon * a + b * (1 - epsilon)


def train_model(model, lossfunc, optimizer, scheduler, num_epochs=10):
    start_time = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    train_acc = []
    valid_acc = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for idx, data in enumerate(dataloders[phase]):
                if idx%50 ==0:
                    print("idx", idx)
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = lossfunc(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)
            if phase == 'train':
                epoch_loss = running_loss / train_img_size
                epoch_acc = running_corrects / train_img_size
            else:
                epoch_loss = running_loss / val_img_size
                epoch_acc = running_corrects / val_img_size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val':
                valid_acc.append(epoch_acc)
            else:
                train_acc.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        # 这里使用了学习率调整策略
        scheduler.step(valid_acc[-1])
    elapsed_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_acc, valid_acc

# 输入图片的尺寸
size = 224 #320#286#224
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
    #transforms.Resize(286),
    transforms.Resize((size,size)),
    # transforms.CenterCrop((320,320)),# transforms.CenterCrop((256,256)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    normalize
])
 
valid_transforms = transforms.Compose([
    transforms.Resize((224,224)),#transforms.Resize((256,256)),
    transforms.ToTensor(),
    normalize
])
# 目录文件
data_dir = './../sorted_data'
train_dir = os.path.join(data_dir,"train")
valid_dir = os.path.join(data_dir,"val")
train_img_size = len(glob.glob(train_dir+"/*/*.jpg"))
val_img_size = len(glob.glob(valid_dir+"/*/*.jpg"))
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
print("train_data.class_to_idx",train_data.class_to_idx)
TRAIN_BATCH_SIZE=32
trainloader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=TRAIN_BATCH_SIZE)
data_loader = [trainloader, validloader]

dataloders = {x:  data_loader[i] for i,x in enumerate(['train', 'val']) }
use_gpu = torch.cuda.is_available()
print("use_gpu",use_gpu)
# 构建模型
# model_ft = models.resnext50_32x4d(pretrained=True)#models.resnet50(pretrained=False)
model_ft = models.resnet50(pretrained=False)#models.resnet50(pretrained=False)
model_ft.load_state_dict(torch.load('./resnet50.pth'))
# num_ftrs = 2048
num_ftrs = model_ft.fc.in_features
# 可以如下在全连接层前面加入dropout来防止过拟合
model_ft.fc = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(num_ftrs, 4)
)
# model_ft.load_state_dict(torch.load('./weights/resnet50_resize224.pth'))
# model_ft.fc=nn.Linear(num_ftrs, 10)
# model_ft.load_state_dict(torch.load('./weights/resnext50_resize224_RandomAffine.pth'))
if use_gpu:
    model_ft = model_ft.cuda()
# define loss function
lossfunc = nn.CrossEntropyLoss()
# lossfunc = LabelSmoothingCrossEntropy()
# 这里直接训练整个网络，也可以像原来baseline先fc后解冻整个网络
parameters = list(model_ft.parameters())
optimizer_ft = optim.SGD(parameters, lr=0.001, momentum=0.9, nesterov=True)
# 使用ReduceLROnPlateau学习调度器，如果三个epoch准确率没有提升，则减少学习率
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft,mode='max',patience=3,verbose=True)
model_ft,train_acc,valid_acc = train_model(model=model_ft,
                           lossfunc=lossfunc,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=20)

# acc 曲线
# %matplotlib inline
import matplotlib.pylab as plt
plt.plot(train_acc,label="train")
plt.plot(valid_acc,label='valid')
plt.legend()
plt.plot()
"""将训练好的模型保存下来。"""
# torch.save(model_ft.state_dict(), './weights/resnet50_resize224.pth')
torch.save(model_ft, './weights/resnet50_resize224.pth')
