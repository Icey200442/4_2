# 目录结构

data/: 包含数据集读取 (dataset.py) 与预处理逻辑 (transform.py)。
models/: 模型定义，基于 timm 构建 ConvNext 类。
main.py / train.py: 分布式训练启动脚本与核心训练器。
test.py / infer.py: 模型性能评估与样本推理。

# 数据准备与路径填写

为了运行此项目，您需要在以下脚本中配置您的本地数据路径：

训练数据配置 (main.py)
在 main.py 的 Config 类中填写您的 JSON 索引路径：

测试指标配置 (test.py)
在脚本底部的 __main__ 入口处填写测试集和模型权重路径： 

    Python
    if __name__ == "__main__":
        evaluate_metrics(
            model_path="/您的路径/best_baseline.pth",
            test_json="/您的路径/test.json" 
        )
    
    Python
    if __name__ == "__main__":
        CKPT = "/您的路径/best_baseline.pth"
        IMG_DIR = "/您的路径/images_to_predict/"
    batch_inference(CKPT, IMG_DIR, batch_size=64)
    
# 运行步骤

第一步：环境安装
确保安装了 torch, timm, albumentations 等依赖。

第二步：启动分布式训练
本项目使用 PyTorch DDP 进行多卡加速。请在终端运行：

Bash

***nproc_per_node 代表使用的 GPU 数量***

torchrun --nproc_per_node=2 main.py
第三步：性能评估
训练完成后，运行测试脚本获取 Accuracy, F1-Score, AUC 等指标：

Bash
python test.py

注：骨干网络: 使用 timm 的 convnext_small 提取特征。

数据json文件包括path和label，path为图片位置，label为图片真假（0为非篡改，1为篡改）