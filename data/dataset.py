import json
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from data.transform import MyTransform
import torch
class MyDataset(Dataset): # 
    def __init__(self, json_path, image_size=224):
        """
        Args:
            json_path: Phase 1 准备好的公共数据集索引文件 (e.g., train.json)
            image_size: 适配 ConvNext 的输入尺寸 (224)
            transform: 数据增强与归一化逻辑
        """
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)
        
        self.image_size = image_size
        self.transform = MyTransform(image_size=image_size)

    def __len__(self):
      return len(self.data_list)

    def __getitem__(self,index):
      #获取当前样本元数据
      item = self.data_list[index]
      image_path = item['path']
      label = item['label'] # 0为真，1为假

      #读取图像
      image = Image.open(image_path).convert("RGB")
      
      #3：数据转换
      image = image.resize((self.image_size,self.image_size))
      image = np.array(image)

      image = self.transform(image)

      output = {
        "image":image,
        "label":torch.tensor(label,dtype=torch.float),
        'image_path':image_path
      }

      return output

#test