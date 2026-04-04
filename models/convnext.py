import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class ConvNext(nn.Module):
  def __init__(self,num_classes=1,pretrained=True):
    '''
    专门用于分类任务设计的ConvNext模型
    Args:
      pretrained(bool):是否加载预训练权重
      num_classes(int):分类数量,二分类任务设为1,配合Sigmoid输出概率
    '''
    #加载骨干网络
    #使用timm加载convnext_small变体
    super(ConvNext,self).__init__()
    self.model = timm.create_model(
      'convnext_small',
      pretrained=pretrained,
      num_classes=num_classes
    )

    #获取特征提取部分
    #timm的forward_features会在全局池化层之前就停止
    self.backbone = self.model.forward_features

    #定义分类头
    #获取特征维度
    out_channels = self.model.num_features

    self.head = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),#全局平均池化
      nn.Flatten(),#展平为向量[B,768]
      nn.Linear(out_channels,num_classes)#映射到最终分类[B,1]
    )

  def forward(self, image, label=None):
    '''
    前向传播
    Args:
        image:输入张量[Batch,3,224,224]
        label:真是标签,用于训练
    '''
    #特征提取
    features = self.backbone(image)

    #得到分类结果
    logits = self.head(features)

    if logits.shape[1] == 1:
      logits = logits.squeeze(dim=1)

    output = {
      "pred":torch.sigmoid(logits),
    }   

    if label is not None:
      loss=F.binary_cross_entropy_with_logits(logits, label.float())
      output["backward_loss"] = loss
    # output = {
    #   "pred":torch.sigmoid(logits),
    #   "backward_loss":loss,
    #   "visual_loss":{
    #     "combined_loss":loss
    #   }
    # }
    #test
    return output