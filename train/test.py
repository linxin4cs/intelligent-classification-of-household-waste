"""### 加载依赖"""

from __future__ import print_function, division
import glob
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pylab as plt
#
plt.rcParams['font.sans-serif'] = ['SimHei']
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms


idx_to_classes={0:'其他垃圾_烟蒂', 1:'厨余垃圾_水果果皮', 2:'可回收物_金属食品罐', 3:'有害垃圾_干电池'}
# 加载模型
# model_ft = models.resnet50(pretrained=False)
# num_ftrs = model_ft.fc.in_features
# # 可以如下在全连接层前面加入dropout来防止过拟合
# model_ft.fc = nn.Sequential(
#     # nn.Dropout(0.1),
#     nn.Linear(num_ftrs, 4)
# )
model_ft=torch.load('./weights/resnet50_resize224.pth')
model_ft.eval()
if torch.cuda.is_available():
    model_ft = model_ft.cuda()

data_dir =  './../garbage_datasets/test_data/*.jpg'
imgs_path=glob.glob(data_dir)
imgs_total=len(imgs_path)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize((224,224)),#transforms.Resize((256,256)),
    transforms.ToTensor(),
    normalize
])

def classify(model,img):
    outputs=model(Variable(img)).cuda()
    pred,ind=torch.max(F.softmax(outputs,dim=1).data,1)
    return pred.item(),ind.item()


for i,img_path in enumerate(imgs_path):
    pil_img=Image.open(img_path).convert("RGB")
    img=preprocess(pil_img)
    img=img.unsqueeze(0).cuda()
    model_ft=model_ft.cuda()
    pred,ind=classify(model_ft,img)
    print("The probablity is :", pred)
    # print("The image path is ",img_path)
    print("The class belongs to :", idx_to_classes[ind])
    plt.imshow(np.asarray(pil_img))
    plt.title("识别结果："+idx_to_classes[ind], fontsize=20)
    plt.show()
