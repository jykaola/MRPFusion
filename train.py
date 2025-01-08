import datetime
import os
import random
import time
import pynvml  # NVIDIA Management Library (NVML) 的Python接口，用于监控GPU的状态，如内存使用情况
import numpy as np
import torch
import logging
from prettytable import PrettyTable  # 用于生成美观的ASCII表格的库
from MRPFusion import MRPFusion
from logger import setup_logger  # 自定义的日志配置函数。
from torch.autograd import Variable  # 早期版本的PyTorch中用于自动微分的变量类
from torch.utils.data import DataLoader
from loss import Fusionloss  # 定制的损失函数
from dataset_util import Fusion_dataset  # 自定义的数据集类，用于加载红外和可见光图像数据。
from evaluation import eval_multi_method  # 用于评估模型性能的函数，可能包括多种度量标准
from args import get_train_args  # 解析命令行参数，用于获取训练相关的参数设置
from torch.utils.tensorboard import SummaryWriter
from my_util import ColorSpaceTransform, RunningTime  # 处理红外和可见光图像，确保它们在融合前处于相同的色彩空间，用于监控和报告融合过程或其他相关步骤的执行时间
from test import test
from tqdm import tqdm

gpu_tamp = 78  # 控制GPU温度.
weight = [11, 40, 2, 20]   # 强度，梯度，SSIM，参数化平衡损失#11, 40, 2, 20    11, 40, 4, 20加训
eval_flag = False  # 是否要指标评测



# 设置随机种子，以确保在训练或推理过程中能够重现相同的结果
def init_seeds(args, seed=1):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    # cudnn seed 0 表示当随机种子为 0 时，程序会启用更慢但更可复现的设置（deterministic mode）。如果种子不为 0，程序会启用更快但不完全可复现的设置。
    import torch.backends.cudnn as cudnn  # NVIDIA cuDNN库的PyTorch接口，用于加速深度神经网络的训练
    random.seed(seed)  # 全局随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.gpu >= 0:  # GPU可用
        torch.cuda.manual_seed(seed)  # 设置 CUDA 设备的随机数生成器种子
    # cuDNN 的可重复性与性能设置
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 1 else (True, False)
    # seed=0时，benchmark=False,deterministic=True
    # seed=1时，benchmark=True,deterministic=False
    '''
    cudnn.benchmark：
    cudnn.benchmark：如果设置为 True，cuDNN 会在运行时为特定的输入尺寸和卷积算法选择最优的计算方式，从而加快训练速度，但这会导致不可重复的结果。
    如果设置为 False，程序速度会变慢，但结果是可重复的。
    cudnn.deterministic
    当设置为 True 时，cuDNN 将强制使用确定性的算法，保证每次运行的结果相同，但速度较慢。

    当种子为 0 时，程序会启用 cudnn.deterministic，确保结果完全可复现，同时关闭 cudnn.benchmark 以避免引入不确定性。
    但如果种子不为 0，则启用 cudnn.benchmark，允许使用更快的非确定性算法来提高训练速度
    '''


def train():
    # 设置一个PrettyTable对象，它用于以表格的形式展示训练过程中的评估指标
    # 当eval_flag为真时，创建一个PrettyTable实例，并定义了表格的列标题
    if eval_flag:
        table = PrettyTable(['Epoch', 'EN', 'MI', 'SF', 'AG', 'SD', 'CC', 'SCD',
                             'VIF', 'MSE', 'PSNR', 'Qabf', 'Nabf', 'SSIM', 'MS_SSIM'])
    start_time = time.time()
    # 一、初始化配置
    # 1.1 配置超参
    args = get_train_args()  # 可以通过 args 对象访问各个参数，例如
    init_seeds(args)
    # 1.2 配置设备
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    # print('Device is {}'.format(device))
    # 1.3 加载数据集
    train_dataset = Fusion_dataset('train', args.resize_flag, args.ir_path, args.vi_path)
    # print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=True,
        drop_last=False
    )

    train_loader.n_iter = len(train_loader)#相当于一轮的样本数/一个批次数量 1083/6=181
    # 1.4 加载网络模型
    ModelNet = easyFusion()
    if args.gpu >= 0:
        ModelNet.to(device)
    # 1.5 配置损失函数
    model_loss = Fusionloss(weight)  # 返回的一个元组，五个损失值：(loss_total)、 (loss_in)、 (loss_grad)、 (loss_ssim)、(loss_tra)
    # 1.6 配置优化器
    optimizer = torch.optim.Adam(ModelNet.parameters(), lr=args.learning_rate)
    # 1.7 添加tensorboard
    writer = SummaryWriter("./logs_train")
    # 1.8 配置logs消息日志
    log_path = './logs'
    logger = logging.getLogger()
    setup_logger(log_path)
    # 1.9 GPU温度监控
    pynvml.nvmlInit()  # 初始化
    # 获取GPU i的handle，后续通过handle来处理
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # 二、加载现有的最佳模型，继续训练 修改部分
    # best_loss = float('Inf')  # 初始化最好损失
    # best_model_file = r"C:\Users\Administrator\Desktop\chatpoint\FusionModel10best.pth"
    # start_epoch = 100  # 从第100轮开始
    # if os.path.exists(best_model_file):
    #     ModelNet.load_state_dict(torch.load(best_model_file))  # 加载最佳模型
    #     logger.info(f"Loaded model from {best_model_file}, resuming from epoch {start_epoch}")

    # 初始化最好损失
    best_loss = float('inf')  # 设置为正无穷大以确保第一次保存模型
    best_model_file = "checkpoint"  # 记录最好的模型权重文件路径

    # 二、训练融合网络
    runtime = RunningTime()  # 返回的是一个包含三个值的元组。eta: 估计的剩余训练时间 this_time: 当前训练周期的持续时间 now_it: 当前已经训练的样本数量
    for epo in range(0, args.epoch):
        logger.info("--------------------------第{}轮训练--------------------------".format(epo + 1))

        # if epo < int(args.epoch // 2):  # epoch到总的一半的时候，学习率会随着epoch的增大而减小
        #     lr = args.learning_rate
        # else:
        #     lr = args.learning_rate * (args.epoch - epo) / (args.epoch - args.epoch // 2)
        if epo < (args.epoch // 2):  # epoch到总的一半的时候，学习率从0.001变成0.0001
            lr = args.learning_rate
        else:
            lr=0.1 * args.learning_rate
        # 更新lr 新学习率 lr 实际应用到优化器中
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 开始训练 训练模式
        ModelNet.train()

        total_loss_fusion = 0  # 用于累加每个批次的损失

        # it 是当前数据批次的索引，从 0 开始递增
        for it, (image_vis, image_ir, name, h, w) in enumerate(tqdm(train_loader, desc=f'Epoch {epo + 1}/{args.epoch}', unit='batch')):
            # 图像数据转换为 Variable 对象，并移到指定的设备（CPU 或 GPU）上进行计算
            # Variable 对象是用于封装张量（Tensor）的一个类，最初是为了支持自动求导和梯度计算。Variable 对象现在已经被合并到 Tensor 中
            image_vis = Variable(image_vis).to(device)
            # RGB 颜色空间转换为 YCrCb 颜色空间，并提取 Y 通道（亮度信息）
            image_vis_ycrcb = ColorSpaceTransform('RGB2YCrCb', image_vis)
            image_vis_y = image_vis_ycrcb[:, :1, :, :]
            image_ir = Variable(image_ir).to(device)
            optimizer.zero_grad()
            # forward
            logits = ModelNet(image_vis_y, image_ir)
            # loss
            loss_fusion, loss_in, loss_grad, loss_ssim, loss_tra = model_loss(image_vis_ycrcb, image_ir, logits)
            loss_fusion.backward()  # 反向传播 计算损失函数的梯度
            optimizer.step()

            total_loss_fusion += loss_fusion.item()  # 累加每个批次的loss

            # print loss 训练过程中输出当前训练状态，包括损失、学习率、时间估计等，并将这些信息记录到日志文件和 TensorBoard 中
            eta, this_time, now_it = runtime.runtime(epo, it, train_loader.n_iter, args.epoch)
            if now_it % 100 == 0:
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_fusion:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'loss_ssim: {loss_ssim:.4f}',
                        'loss_tra: {loss_tra:.4f}',
                        'lr: {lr:.4f}',
                        'time: {time:.2f}',
                        'eta: {eta}'
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * args.epoch,
                    loss_fusion=loss_fusion.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_ssim=loss_ssim.item(),
                    loss_tra=loss_tra.item(),
                    lr=lr,
                    time=this_time,
                    eta=eta
                )
                logger.info(msg)
                writer.add_scalar("train_loss", loss_fusion.item(), now_it)

                # GPU温度检测
                gpuTemperature = pynvml.nvmlDeviceGetTemperature(handle, 0)  # 读取温度
                if gpuTemperature >= gpu_tamp:
                    print('GPU温度超过{}℃，开始降温！'.format(gpu_tamp))
                    time.sleep(5)  # 延时，让GPU温度没那么高
        # 计算当前 epoch 的平均损失
        avg_loss_fusion = total_loss_fusion / len(train_loader)
        print("第{}轮的总损失为:{},平均损失为:{}".format(epo + 1, total_loss_fusion, avg_loss_fusion))
        # 三、保存模型权重
        if (epo + 1) >= 1:
            fusion_model_file = os.path.join(args.fusion_model_path,
                                             'FusionModel{}.pth'.format(epo + 1))  # 构建保存模型权重的文件路径
            torch.save(ModelNet.state_dict(), fusion_model_file)  # 保存当前模型的权重到指定的文件。
            logger.info("Fusion Model Save to: {}".format(fusion_model_file))
            if avg_loss_fusion < best_loss:
                best_loss = avg_loss_fusion
                best_model_file = os.path.join(args.fusion_model_path, 'BestFusionModel.pth')
                torch.save(ModelNet.state_dict(), best_model_file)
                logger.info(f"Best Fusion Model Saved to: {best_model_file} with loss: {best_loss:.4f}")

            if eval_flag:
                test((epo + 1), 'TNO')
                EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = \
                    eval_multi_method(easy_flag=easy_flag)
                val_list = [str((epo + 1)), round(EN, 4), round(MI, 4), round(SF, 4), round(AG, 4), round(SD, 4),
                            round(CC, 4), round(SCD, 4), round(VIF, 4), round(MSE, 4), round(PSNR, 4),
                            round(Qabf, 4), round(Nabf, 4), round(SSIM, 4), round(MS_SSIM, 4)]
                table.add_row(val_list)
                logger.info(val_list)
            elif (epo + 1) == args.epoch:
                # test((epo + 1), 'TNO')
                test((epo + 1), 'MSRS')

        # if (epo + 1) >= args.epoch // 2:
        #     test((epo + 1), 'MSRS')

    end_time = time.time()
    all_time = int(end_time - start_time)
    eta = str(datetime.timedelta(seconds=all_time))
    logger.info('\n')
    logger.info('All train time is {}'.format(eta))
    if eval_flag:
        table.add_row(['SeAFusion', 7.1335, 2.833, 12.2525, 4.9803, 44.2436, 0.4819,  # 参考对比指标
                       1.7281, 0.7042, 0.059, 61.3917, 0.4879, 0.0807, 0.963, 0.9716])
        logger.info(table.get_string())  # 打印评价指标
    logger.info('weight = {}'.format(weight))


if __name__ == '__main__':
    train()

