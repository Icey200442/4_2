import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class MyTransform:
    def __init__(self, image_size=224):
        """
        封装标准预处理流程
        Args:
            image_size: 适配 ConvNext 的输入尺寸
        """
        self.transform = A.Compose([
            # 1. 尺寸调整：确保 Batch 训练时张量形状一致
            A.Resize(image_size, image_size),
            
            # 2. 归一化：使用 ImageNet 的均值和标准差 (ConvNext 预训练权重的标准)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            # 3. 核心：将 Numpy 数组 [H, W, C] 转为 PyTorch Tensor [C, H, W]
            ToTensorV2()
        ])

    def __call__(self, image):
        """
        使对象可调用。输入 image 必须是 Numpy 数组。
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # 返回字典中提取出来的 Tensor
        return self.transform(image=image)['image']
        #test