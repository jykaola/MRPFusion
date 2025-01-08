# coding:utf-8
import os
import shutil
import time
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
import evaluation
from dataset_util import Fusion_dataset
from args import get_test_args
from MRPFusion import MRPFusion, model_deploy
from my_util import ColorSpaceTransform, ImageProcessing, algorithm_runtime
from torchvision import transforms
import numpy as np


# tensorboard: tensorboard --logdir=logs_feature
# 如果出现阴间绿图，一定是权重没加载！！！


num = 10 # 从第几个权重开始往后生成图片
loop_num = 1  # 循环几组
num_step = 1  # 相差步长

#测试模型的性能并保存测试输出的融合图像
def test(num, dataset='test', deploy_flag=True):
    # 一、初始化配置
    # 1.1 配置超参
    args = get_test_args()
    # 1.2 配置设备
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    # 1.3 加载模型
    ModelNet = MRPFusion()
    if args.gpu >= 0:
        ModelNet.to(device)
    # 1.4 加载模型权重
    # fusion_model_file = os.path.join(args.fusion_model_path, 'FusionModel{}.pth'.format(num))
    #加载最佳权重
    # fusion_model_file = os.path.join(args.fusion_model_+path, 'BestFusionModel.pth')
    fusion_model_file = os.path.join(args.fusion_model_path, r"D:\project\pythonTh_poject\MRPFusion\checkpoint\BestFusionModel.pth")
    ModelNet.load_state_dict(torch.load(fusion_model_file))
    if deploy_flag:
        ModelNet = model_deploy(ModelNet)
    # 1.5 加载数据集
    if dataset == 'MSRS':
        ir_path = args.ir_path1
        vi_path = args.vi_path1
        fused_path = args.fused_path1
    elif dataset == 'TNO':
        ir_path = args.ir_path2
        vi_path = args.vi_path2
        fused_path = args.fused_path2
    elif dataset == 'RoadScene':
        ir_path=args.ir_path3
        vi_path = args.vi_path3
        fused_path = args.fused_path3
    elif dataset == 'M3FD':
        ir_path = args.ir_path4
        vi_path = args.vi_path4
        fused_path = args.fused_path4
    elif dataset == 'dataset':
        ir_path = args.ir_path5
        vi_path = args.vi_path5
        fused_path = args.fused_path5



    test_dataset = Fusion_dataset('test', args.resize_flag, ir_path=ir_path, vi_path=vi_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        pin_memory=True,
        drop_last=False
    )
    if not os.path.exists(fused_path):  # 如果没有路径文件夹，就创建文件夹
        os.makedirs(fused_path)
        print('Making fused_path {}'.format(fused_path))

    # 二、融合步骤
    ModelNet.eval()
    x = torch.randn(1, 1, 64, 64).to(device)
    _ = ModelNet(x, x)  # 提前跑个数据，不然第一个的运行时间不是真正的时间
    AllRunTime = []
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir, name, h, w) in enumerate(test_loader):
            # 2.1 提取dataset
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)

            if args.gpu >= 0:
                images_vis = images_vis.to(device)#[1,3,768,1024]
                images_ir = images_ir.to(device)#[1,1,768,1024]
            # 2.2 色域空间转换——RGB2YCrCb
            images_vis_ycrcb = ColorSpaceTransform('RGB2YCrCb', images_vis)
            image_vis_y = images_vis_ycrcb[:, :1, :, :]#从images_vis_ycrcb中提取第一个通道 即Y 通道。Y 通道代表亮度信息
            # 2.3 融合图像
            start_time = time.time()
            logits = ModelNet(image_vis_y, images_ir)#[1,1,768,1024]
            # 2.4 拼回完整的YCrCb图像（融合图像作为Y通道，Cb、Cr通道用原来的）
            fusion_ycrcb = torch.cat((logits, images_vis_ycrcb[:, 1:2, :, :], images_vis_ycrcb[:, 2:, :, :]), dim=1)
            # 2.5 色域空间转换——YCrCb2RGB
            fusion_image = ColorSpaceTransform('YCrCb2RGB', fusion_ycrcb)
            # 2.6 图像处理——对像素进行归一化等处理
            fused_image = ImageProcessing(fusion_image)
            # 2.7 保存图片
            image = fused_image[0, :, :, :]
            image = Image.fromarray(image)#将 NumPy 数组转换为 PIL 图像
            if 'TNO' in fused_path:
                save_path = os.path.join(fused_path, '{}'.format(name[0]))
            elif 'RoadScene' in fused_path:
                save_path = os.path.join(fused_path, '{}'.format(name[0]))
            elif 'MSRS' in fused_path:
                save_path = os.path.join(fused_path, '{}'.format(name[0]))
            elif 'M3FD' in fused_path:
                save_path = os.path.join(fused_path, '{}'.format(name[0]))
            else:
                save_path = os.path.join(fused_path, '{}_{}'.format(num, name[0]))
                # image = image.resize((w, h))
            image.save(save_path)
            run_time = time.time() - start_time
            test_bar.set_description("Fusion: {} | Run time: {:.04f}s".format(save_path, run_time))
            test_bar.update(1)
            # AllRunTime.append(run_time)

        # 2.8 运行时间计算
        # algorithm_runtime(AllRunTime)


#################################################################################
# 删除文件函数：del_file
# 删除目录下的文件，用于删除训练过程的图片
#################################################################################
def del_file(path):
    if not os.listdir(path):
        print('目录为空！')
    else:
        for i in os.listdir(path):
            path_file = os.path.join(path, i)  # 取文件绝对路径
            print("Remove ", path_file)
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                del_file(path_file)
                shutil.rmtree(path_file)


if __name__ == '__main__':
    # del_file('./results')

    # for i in range(0, loop_num):
    #     test(num)
    #     num = num - num_step

    # num = 6
    for i in range(0, loop_num):
        test(num, 'TNO')   # TNO
        test(num, 'MSRS')
        # test(num, 'RoadScene')
        test(num, 'M3FD')
        num = num - num_step
