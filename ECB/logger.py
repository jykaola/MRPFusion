import os.path as osp#用于处理文件路径的模块
import time
import logging#Python 的标准日志记录库
import torch.distributed as dist#PyTorch 的分布式训练库，用于检查分布式训练的状态
import os

#logpth 参数: 这是日志文件存储的路径
def setup_logger(logpth):
    logfile = 'FusionNet-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)#日志路径 logpth 和文件名结合起来
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    #%(levelname)s 表示日志级别（如 INFO、ERROR）%(filename)s 是文件名，%(lineno)d 是行号，%(message)s 是实际的日志消息。
    log_level = logging.INFO#INFO 是一个日志级别 普通的运行信息，用于记录程序的正常运行状态和重要的事件
    #检查分布式训练是否已初始化并且当前进程不是 rank 0，则将日志级别设置为 ERROR。这意味着非主进程只记录错误信息，以减少日志量。
    if dist.is_initialized() and not dist.get_rank() == 0:
        log_level = logging.ERROR
    #设置日志记录的级别、格式和文件路径
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    #创建一个流处理器，将日志消息输出到控制台。这确保了日志消息不仅被写入文件，还会实时显示在控制台上
    logging.root.addHandler(logging.StreamHandler())
