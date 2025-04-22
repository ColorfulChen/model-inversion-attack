# Model Inversion Attack Implementation

## 项目背景
基于深度学习的模型反演攻击实现，通过模型输出来重构输入数据。本实现针对AT&T人脸数据集进行攻击实验。

## 环境依赖
```bash
pip install -r requirements.txt
```

## 命令行参数说明
```python
python main.py model-inversion attack-dataset-cmd --dataset_path dataset/data_pgm --output_dir attack_results --iterations 50 --loss_function crossEntropy
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|-----|
| --dataset_path | string | dataset/data_pgm | 包含PGM格式数据集的目录路径 |
| --output_dir | string | attack_results | 攻击结果输出目录 |
| --iterations | integer | 50 | 攻击迭代次数 |
| --loss_function | [crossEntropy, softmax] | crossEntropy | 使用的损失函数类型 |

## 结果示例
攻击结果将生成对比图，包含原始图像和重构图像：
```
attack_results/
└── s01/
    ├── attack_1.png
    ├── attack_2.png
    └── ...
```

## 许可证
MIT License