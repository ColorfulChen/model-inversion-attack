import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from PIL import Image
from ModelInversion.target_model import TargetModel, train_target_model

def load_pgm_image(file_path):
    """加载PGM格式图片"""
    with open(file_path, 'rb') as f:
        # 读取PGM文件头
        header = f.readline().decode('utf-8').strip()
        if header != 'P5':
            raise ValueError('Not a valid PGM file')
        
        # 读取尺寸信息
        dimensions = f.readline().decode('utf-8').strip()
        while dimensions.startswith('#'):
            dimensions = f.readline().decode('utf-8').strip()
        width, height = map(int, dimensions.split())
        
        # 读取最大灰度值
        max_val = int(f.readline().decode('utf-8').strip())
        
        # 读取图像数据
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        image = image_data.reshape((height, width))
        
        return image

def mi_face(target_class, model, iterations, gradient_step_size, loss_function):
    """执行模型逆向攻击"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # 初始化随机输入
    x = torch.randn(1, 112 * 92, requires_grad=True, device=device)
    
    # 定义优化器
    optimizer = optim.Adam([x], lr=gradient_step_size)
    
    # 目标输出（one-hot编码）
    target = torch.zeros(1, 40, device=device)
    target[0, target_class] = 1
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # 前向传播
        output, _ = model(x)
        
        # 计算损失
        if loss_function == "crossEntropy":
            loss = nn.CrossEntropyLoss()(output, target)
        else:  # softmax
            loss = nn.MSELoss()(torch.softmax(output, dim=1), target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 将图像值限制在合理范围内
        with torch.no_grad():
            x.data = torch.clamp(x.data, -1, 1)
    
    return x

def attack_dataset(dataset_path, output_dir='attack_results', iterations=50, loss_function="crossEntropy"):
    """对数据集中的所有图片执行攻击"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载或训练目标模型
    print("Loading target model...")
    model = TargetModel(112 * 92, 40)
    model_path = os.path.join(os.path.dirname(__file__), 'ModelInversion', 'atnt-mlp-model.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Model not found, training new model...")
        train_target_model(30)  # 训练30个epoch
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    print(f"Found {len(subfolders)} subfolders")
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(dataset_path, subfolder)
        print(f"\nProcessing subfolder: {subfolder}")
        
        # 获取子文件夹中的所有PGM文件
        pgm_files = [f for f in os.listdir(subfolder_path) if f.endswith('.pgm')]
        print(f"Found {len(pgm_files)} PGM files in {subfolder}")
        
        # 为每个子文件夹创建输出目录
        subfolder_output_dir = os.path.join(output_dir, subfolder)
        if not os.path.exists(subfolder_output_dir):
            os.makedirs(subfolder_output_dir)
        
        for pgm_file in pgm_files:
            print(f"Processing {pgm_file}...")
            try:
                # 加载原始图片
                image_path = os.path.join(subfolder_path, pgm_file)
                original_image = load_pgm_image(image_path)
                
                # 获取目标类别（从文件名中提取）
                target_class = int(pgm_file.split('.')[0]) - 1  # 转换为0-based索引
                
                # 执行攻击
                reconstructed = mi_face(
                    target_class,
                    model,
                    iterations=iterations,
                    gradient_step_size=0.1,
                    loss_function=loss_function
                )
                
                # 保存结果
                save_path = os.path.join(subfolder_output_dir, f'attack_{pgm_file.replace(".pgm", ".png")}')
                
                # 可视化结果
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(original_image, cmap='gray')
                plt.title('Original')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                reconstructed_image = reconstructed.detach().cpu().numpy().reshape(112, 92)
                plt.imshow(reconstructed_image, cmap='gray')
                plt.title('Reconstructed')
                plt.axis('off')
                
                plt.suptitle(f'Model Inversion Attack - Class {target_class + 1}')
                plt.savefig(save_path)
                plt.close()
                
                print(f"Results saved to {save_path}")
                
            except Exception as e:
                print(f"Error processing {pgm_file}: {str(e)}")
                continue

if __name__ == "__main__":
    # 示例使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载预训练模型（这里使用ResNet18作为示例）
    model = torchvision.models.resnet18(pretrained=True)
    model = model.to(device)
    
    # 创建攻击实例
    attack = ModelInversionAttack(
        target_model=model,
        num_classes=1000,  # ImageNet类别数
        input_shape=(3, 224, 224),
        device=device
    )
    
    # 对每个类别执行攻击
    for target_class in range(10):  # 这里只攻击前10个类别作为示例
        print(f"Attacking class {target_class}")
        final_image, reconstructed_images = attack.attack(target_class)
        
        # 可视化结果
        save_path = f"attack_results_class_{target_class}.png"
        visualize_attack(reconstructed_images + [final_image], target_class, save_path) 