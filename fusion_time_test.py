import torch
import time
from MRPFusion import MRPFusion, model_deploy  # 假设你的模型定义在这个文件中
from PIL import Image
import os
import numpy as np

def load_model_and_calculate_fusion_time(model, ir_image, vis_image, device, n=100):
    # 第一次运行以消除初始化开销
    for _ in range(10):
        _ = model(ir_image, vis_image)

    # 计算融合时间
    start_time = time.time()
    for _ in range(n):
        fused_image = model(ir_image, vis_image)
    end_time = time.time()

    # 计算平均时间
    avg_time = (end_time - start_time) / n
    return avg_time

def preprocess_image(image_path, device):
    # 读取图像
    image = Image.open(image_path).convert('L')  # 转换为灰度图像
    # 转换为NumPy数组
    image_np = np.array(image, dtype=np.float32)
    # 归一化到 [0, 1] 范围
    image_np = image_np / 255.0
    # 转换为Tensor
    image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(device)
    return image_tensor

def test_fusion_time_on_tno_dataset(model_path, ir_dir, vis_dir, device, n=100,deploy_flag=True):
    # 加载模型
    ModelNet = MRPFusion().to(device)
    ModelNet.load_state_dict(torch.load(model_path))

    if deploy_flag:
        ModelNet = model_deploy(ModelNet)
    ModelNet.eval()
    # 获取所有图像文件名
    ir_images = sorted(os.listdir(ir_dir))
    vis_images = sorted(os.listdir(vis_dir))

    # 确保两个目录中的图像数量相同
    assert len(ir_images) == len(vis_images), "Number of IR and VIS images must be the same"

    # 测试21对图像
    num_pairs = min(21, len(ir_images))
    total_time = 0.0

    for i in range(num_pairs):
        ir_image_path = os.path.join(ir_dir, ir_images[i])
        vis_image_path = os.path.join(vis_dir, vis_images[i])

        # 预处理图像
        ir_image = preprocess_image(ir_image_path, device)
        vis_image = preprocess_image(vis_image_path, device)

        # 计算融合时间
        avg_time = load_model_and_calculate_fusion_time(ModelNet, ir_image, vis_image, device, n=n)
        total_time += avg_time

        print(f'Pair {i+1}: Average fusion time: {avg_time:.4f} seconds per iteration')

    # 计算平均时间
    average_total_time = total_time / num_pairs
    print(f'Average fusion time for all pairs: {average_total_time:.4f} seconds per iteration')

# 设定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型权重路径
model_path = r"C:\Users\Administrator\Desktop\chatpoint\FusionModel10best.pth"

# TNO数据集路径
ir_dir = r"D:\project\pythonTh_poject\TNO\Inf"
vis_dir = r"D:\project\pythonTh_poject\TNO\Vis"

# MSRS数据集路径
# ir_dir = r"D:\project\pythonTh_poject\MSRS\Inf"
# vis_dir = r"D:\project\pythonTh_poject\MSRS\Vis"

# M3FD数据集路径
# ir_dir = r"D:\project\pythonTh_poject\M3FD\Inf"
# vis_dir = r"D:\project\pythonTh_poject\M3FD\Vis"

# 测试融合时间
test_fusion_time_on_tno_dataset(model_path, ir_dir, vis_dir, device,n=1,deploy_flag=True)