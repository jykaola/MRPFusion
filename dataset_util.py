import os
import time
import torch
from natsort import natsorted
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import glob


#用于准备数据路径  返回路径下的文件名列表和每个文件的路径列表
def prepare_data_path(dataset_path):
    #列出目录中的所有文件名和路径按自然排序进行排序
    filenames = natsorted(os.listdir(dataset_path))
    data_dir = dataset_path
    #glob.glob 函数用于查找与模式匹配的所有路径 返回一个列表，列表中包含了所有匹配 bmp 模式的文件和目录的完整路径
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data = natsorted(data)
    return data, filenames


class Fusion_dataset(Dataset):
    def __init__(self, split, resize_flag, ir_path=None, vi_path=None):
        #split: 指定数据集的类型，'train' 或 'test'，用于区分训练集和测试集。
        #resize_flag: 一个标志，指示是否需要对图像进行缩放处理
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'test'], 'split must be "train"|"test"'
        # 为了方便有些模型是需要标签或者别的不同训练集，因此分开写，有用到专用的标签或别的时候就专门去写路径
        self.h = 512
        self.w = 512
        if split == 'train':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            #调用 prepare_data_path 获取可见光图像和红外图像的文件路径和文件名
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.resize_flag = resize_flag
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'test':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.resize_flag = resize_flag
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split == 'train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            if self.resize_flag:
                image_vis = np.array(Image.open(vis_path).resize((320, 240), Image.LANCZOS))
                image_inf = cv2.imread(ir_path, 0)#参数 0 以灰度模式读取图像。
                self.h, self.w = image_inf.shape
                image_inf = cv2.resize(image_inf, (320, 240), interpolation=cv2.INTER_CUBIC)#cv2.INTER_CUBIC 是一种立方插值方法，用于在缩放图像时获得较好的视觉效果
            else:
                image_vis = np.array(Image.open(vis_path))
                image_inf = cv2.imread(ir_path, 0)#参数 0 以灰度模式读取图像
                self.h, self.w = image_inf.shape#记录了红外图像的原始高度和宽度

            #transpose((2, 0, 1)): 从 (H, W, C) 转换为 (C, H, W)
            #标准化处理将图像像素值从 [0, 255] 缩放到 [0.0, 1.0]
            image_vis = (np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0)
            #image_inf 是一个灰度图像，所以在这里的 np.asarray 之后，它将保持 (H, W) 的形状。
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            #通过index获取当前图像的文件名
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
                self.h,
                self.w
            )
        elif self.split == 'test':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            if self.resize_flag:
                image_vis = np.array(Image.open(vis_path).resize((512, 512), Image.LANCZOS))
                image_inf = cv2.imread(ir_path, 0)
                self.h, self.w = image_inf.shape
                image_inf = cv2.resize(image_inf, (512, 512), interpolation=cv2.INTER_CUBIC)
            else:
                image_vis = np.array(Image.open(vis_path))
                image_inf = cv2.imread(ir_path, 0)
                self.h, self.w = image_inf.shape

            image_vis = (np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0)
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
                self.h,
                self.w
            )

    def __len__(self):
        return self.length


#测试功能
# if __name__ == '__main__':
#     data_dir = r"D:\project\pythonTh_poject\OwnFusion\datasets\TNO"
#     Vir_path=data_dir+"\\"+"Vis"
#     Inf_path = data_dir + "\\" + "Inf"
#     data, filenames = prepare_data_path(Vir_path)
#     train_dataset = Fusion_dataset(split="train",resize_flag=False,ir_path=Inf_path,vi_path=Vir_path)
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=1,
#         shuffle=True,
#         num_workers=2,
#         pin_memory=True,
#         drop_last=True,
#     )
#     train_loader.n_iter = len(train_loader)#添加一个新的属性 n_iter,表示批次的数量,注意不是一批图像的数量，而是有几批
#
#
#     for batch_idx, (vis_images, ir_images, names, heights, widths) in enumerate(train_loader):
#         print(f"Batch index: {batch_idx}")
#
#         # 确保 vis_images 和 ir_images 的形状是 (batch_size, channels, height, width)
#         print(f"Visible images shape: {vis_images.shape}")
#         print(f"Infrared images shape: {ir_images.shape}")
#
#         # 遍历批次中的每张图像
#         for i in range(len(vis_images)):
#             height, width = vis_images[i].size(1), vis_images[i].size(2)  # 获取高度和宽度
#             print(f"Image {i}: Height: {height}, Width: {width}")
#
#         # 如果需要访问其他信息，如 names, heights, widths
#         for i in range(len(names)):
#             print(f"Image {i} - Name: {names[i]}, Height: {heights[i]}, Width: {widths[i]}")
