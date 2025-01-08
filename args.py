import argparse#用于命令行参数解析的模块

#解析命令行参数，以配置训练过程的参数
def get_test_args():
    parser = argparse.ArgumentParser(description='Test Image Fusion Model With Pytorch!')

    #add_argument() 方法用于定义命令行参数
    #--batch_size一个长选项名称，用于在命令行中指定参数。在运行脚本时，用户可以通过 --batch_size 后跟一个数值来设置批量大小。
    #-B: 这是一个短选项名称，是 --batch_size 的快捷方式。用户也可以通过 -B 后跟一个数值来设置批量大小
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_worker', '-W', type=int, default=0)  # 加载数据集的CPU线程数
    parser.add_argument('--fusion_model_path', '-S', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\checkpoint')  # 模型权重路径
    parser.add_argument('--resize_flag', '-R', type=bool, default=False)
    parser.add_argument('--logs_path', type=str, default='./logs')  # 日志消息路径

    parser.add_argument('--ir_path1', type=str, default=r"D:\project\pythonTh_poject\OwnFusion\datasets\MSRS\test\Inf")  # 红外图像测试集路径
    parser.add_argument('--vi_path1', type=str, default=r"D:\project\pythonTh_poject\OwnFusion\datasets\MSRS\test\Vis")  # 可见光图像测试集路径
    parser.add_argument('--fused_path1', type=str, default=r"D:\project\pythonTh_poject\OwnFusion\datasets\results\MSRS")  # 融化图像路径

    # parser.add_argument('--ir_path1', type=str,default=r"D:\project\pythonTh_poject\OwnFusion\datasets\MSRS\train\Inf")  # 红外图像训练集路径
    # parser.add_argument('--vi_path1', type=str,default=r"D:\project\pythonTh_poject\OwnFusion\datasets\MSRS\train\Vis")  # 可见光图像训练集路径
    # parser.add_argument('--fused_path1', type=str,default=r"D:\project\pythonTh_poject\OwnFusion\datasets\results\MSRS_train")  # 融化图像路径


    parser.add_argument('--ir_path2', type=str, default=r"D:\project\pythonTh_poject\OwnFusion\datasets\TNO\Inf")  # 红外图像测试集路径
    parser.add_argument('--vi_path2', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\TNO\Vis')  # 可见光图像测试集路径
    parser.add_argument('--fused_path2', type=str, default=r"D:\project\pythonTh_poject\OwnFusion\datasets\results\TNO")  # 融化图像路径

    parser.add_argument('--ir_path3', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\RoadScene\Inf')  # 红外图像测试集路径
    parser.add_argument('--vi_path3', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\RoadScene\Vis')  # 可见光图像测试集路径
    parser.add_argument('--fused_path3', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\results\RoadScene')  # 融化图像路径

    parser.add_argument('--ir_path4', type=str,default=r'D:\project\pythonTh_poject\OwnFusion\datasets\M3FD\Inf')  # 红外图像测试集路径
    parser.add_argument('--vi_path4', type=str,default=r'D:\project\pythonTh_poject\OwnFusion\datasets\M3FD\Vis')  # 可见光图像测试集路径
    parser.add_argument('--fused_path4', type=str, default=r"D:\project\pythonTh_poject\OwnFusion\datasets\results\M3FD")  # 融化图像路径

    # parser.add_argument('--ir_path5', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\M3FD\Inf')  # 红外图像测试集路径
    # parser.add_argument('--vi_path5', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\M3FD\Vis')  # 可见光图像测试集路径
    # parser.add_argument('--fused_path5', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\results\M3FD')  # 融化图像路径

    test_args = parser.parse_args()
    return test_args


def get_train_args():
    #创建一个 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser(description='Train Image Fusion Model With Pytorch!')

    parser.add_argument('--batch_size', '-B', type=int, default=350)
    parser.add_argument('--epoch', '-E', type=int, default=10)
    parser.add_argument('--learning_rate', '-L', type=float, default=1e-3)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--resize_flag', '-R', type=bool, default=False)
    parser.add_argument('--num_worker', '-W', type=int, default=1)  # 加载数据集的CPU线程数
    parser.add_argument('--fusion_model_path', '-S', type=str, default=r'D:\project\pythonTh_poject\MRPFusion\checkpoint')
    parser.add_argument('--dataset_path', type=str, default='./dataset')  # 数据集路径
    parser.add_argument('--logs_path', type=str, default='./logs')  # 日志消息路径
    # parser.add_argument('--ir_path', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\MSRS\train\Inf')  # 红外图像训练集路径
    # parser.add_argument('--vi_path', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\MSRS\train\Vis')  # 可见光图像训练集路径
    parser.add_argument('--ir_path', type=str, default=r'D:\project\pythonTh_poject\MSRS\ir_st')  # 红外图像训练集路径
    parser.add_argument('--vi_path', type=str, default=r'D:\project\pythonTh_poject\MSRS\vi_st')  # 可见光图像训练集路径

    train_args = parser.parse_args()#解析命令行参数并返回一个包含这些参数的对象
    return train_args#返回解析后的参数对象，以便在训练过程中使用


def get_ablation_args():
    parser = argparse.ArgumentParser(description='Test Image Fusion Model With Pytorch!')

    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_worker', '-W', type=int, default=0)  # 加载数据集的CPU线程数
    parser.add_argument('--fusion_model_path', '-S', type=str, default='./checkpoint')  # 模型权重路径
    parser.add_argument('--resize_flag', '-R', type=bool, default=False)
    parser.add_argument('--logs_path', type=str, default='./logs')  # 日志消息路径

    parser.add_argument('--ir_path1', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\TNO\Inf')  # 红外图像测试集路径
    parser.add_argument('--vi_path1', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\TNO\Vis')  # 可见光图像测试集路径
    parser.add_argument('--fused_path1', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\ablation\TNO')  # 融化图像路径

    parser.add_argument('--ir_path2', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\RoadScene\Inf')  # 红外图像测试集路径
    parser.add_argument('--vi_path2', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\RoadScene\Vis')  # 可见光图像测试集路径
    parser.add_argument('--fused_path2', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\ablation\RoadScene')  # 融化图像路径

    parser.add_argument('--ir_path3', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\MSRS\test\Inf')  # 红外图像测试集路径
    parser.add_argument('--vi_path3', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\MSRS\test\Vis')  # 可见光图像测试集路径
    parser.add_argument('--fused_path3', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\ablation\MSRS')  # 融化图像路径

    parser.add_argument('--ir_path4', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\M3FD\Inf')  # 红外图像测试集路径
    parser.add_argument('--vi_path4', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\M3FD\Vis')  # 可见光图像测试集路径
    parser.add_argument('--fused_path4', type=str, default=r'D:\project\pythonTh_poject\OwnFusion\datasets\ablation\M3FD')  # 融化图像路径

    test_args = parser.parse_args()
    return test_args
