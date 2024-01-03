# 对原始数据进行整理
import os
import glob
import shutil

datasets_dir = "./../garbage_datasets/train_data"
sorted_data_dir ="./../sorted_data"
img_paths = glob.glob(datasets_dir+"/*.jpg")
for img_path in img_paths:
    txt_file_path = img_path.replace("jpg","txt")
    with open(txt_file_path,"r") as f:
        cls_name = f.readline().strip().split(",")[1]
        cls_name_dir = os.path.join(sorted_data_dir,cls_name)
        if not os.path.isdir(cls_name_dir):
            os.makedirs(cls_name_dir)
        img_name = img_path.rsplit(os.sep,1)[1]
        new_img_path = os.path.join(cls_name_dir,img_name)
        shutil.copy(img_path,new_img_path)
